[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [string]$ArchivePath = '',

    [ValidatePattern('^[0-9a-f]{64}$')]
    [string]$ArchiveSha256 = '',

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$ReleaseId = '',

    [switch]$ReadState,

    [Parameter(Mandatory = $false)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$ServiceName = '',

    [Parameter(Mandatory = $false)]
    [string]$RestartBatch = '',

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 65535)]
    [int]$ListenPort = 8000
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

function Resolve-WindowsPath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw 'A required Windows path is empty.'
    }

    $NormalizedPath = $Path.Trim().Trim('"').Replace('/', '\')
    if ($NormalizedPath -notmatch '^[A-Za-z]:\\') {
        throw "Expected an absolute Windows path, received: $Path"
    }

    return [System.IO.Path]::GetFullPath($NormalizedPath)
}

$ProjectRoot = Resolve-WindowsPath -Path $ProjectRoot
$TargetApp = Join-Path $ProjectRoot 'app'
$StatePath = Join-Path $ProjectRoot '.nte-deploy-app.json'
$LockPath = Join-Path $ProjectRoot '.nte-deploy-app.lock'
# Staging and rollback stay on the app's volume, so directory moves are atomic.
$WorkRoot = Join-Path (Join-Path $ProjectRoot '.nte-deploy') $ReleaseId
$StageRoot = Join-Path $WorkRoot 'stage'
$IncomingRoot = Join-Path $WorkRoot 'incoming'
$IncomingApp = Join-Path $IncomingRoot 'app'
$BackupApp = Join-Path $WorkRoot 'backup-app'
$FailedApp = Join-Path $WorkRoot 'failed-app'

function Get-ListeningProcessIds {
    param([int]$Port)

    $ProcessIds = @()
    foreach ($Line in (& netstat.exe -ano -p TCP)) {
        $Parts = ($Line.Trim() -split '\s+')
        if (
            $Parts.Length -ge 5 -and
            $Parts[0] -eq 'TCP' -and
            $Parts[1] -match (':' + $Port + '$') -and
            $Parts[3] -eq 'LISTENING'
        ) {
            $ProcessIds += [int]$Parts[4]
        }
    }

    return @($ProcessIds | Select-Object -Unique)
}

function Stop-PortListener {
    param([int]$Port)

    $ProcessIds = @(Get-ListeningProcessIds -Port $Port)
    foreach ($ProcessId in $ProcessIds) {
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
    }

    if ($ProcessIds.Count -gt 0) {
        $Deadline = (Get-Date).AddSeconds(15)
        while ((Get-Date) -lt $Deadline) {
            if (@(Get-ListeningProcessIds -Port $Port).Count -eq 0) {
                return $ProcessIds.Count
            }
            Start-Sleep -Milliseconds 500
        }
        throw "Timed out stopping the process listening on port $Port."
    }

    return 0
}

function Start-WaitressBatch {
    param(
        [string]$BatchPath,
        [string]$WorkingDirectory
    )

    # Win32_Process.Create is provided by the WMI service, so the spawned
    # process is not attached to the short-lived OpenSSH session job.
    # Redirecting stdin from NUL also prevents the batch file's trailing
    # `pause` from leaving a hidden cmd.exe behind after Waitress exits.
    $LogDirectory = Join-Path $WorkingDirectory 'logs'
    if (-not (Test-Path -LiteralPath $LogDirectory -PathType Container)) {
        New-Item -ItemType Directory -Path $LogDirectory | Out-Null
    }
    $LogPath = Join-Path $LogDirectory 'waitress-deploy.log'
    $CommandLine = (
        '"' + $env:ComSpec + '" /d /c call "' +
        $BatchPath + '" >> "' + $LogPath + '" 2>&1 < nul'
    )
    $ProcessClass = [WmiClass]'\\.\root\cimv2:Win32_Process'
    $Result = $ProcessClass.Create($CommandLine, $WorkingDirectory, $null)
    if ($Result.ReturnValue -ne 0) {
        throw (
            "WMI failed to start Waitress batch. ReturnValue=" +
            $Result.ReturnValue
        )
    }
}

function Wait-ForPort {
    param(
        [int]$Port,
        [int]$TimeoutSeconds
    )

    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $StableChecks = 0
    while ((Get-Date) -lt $Deadline) {
        if (@(Get-ListeningProcessIds -Port $Port).Count -gt 0) {
            $StableChecks += 1
            if ($StableChecks -ge 10) {
                return $true
            }
        }
        else {
            $StableChecks = 0
        }
        Start-Sleep -Milliseconds 500
    }

    return $false
}

function Assert-Properties {
    param($Object, [string[]]$Names)
    if ($null -eq $Object) { throw 'Missing deployment metadata.' }
    foreach ($Name in $Names) {
        if ($Object.PSObject.Properties.Name -notcontains $Name) {
            throw "Missing deployment field: $Name"
        }
    }
}

function Assert-AppPath {
    param([string]$Path)
    if (-not $Path.StartsWith('app/', [StringComparison]::Ordinal)) {
        throw "Not an app file: $Path"
    }
    foreach ($Part in $Path.Split('/')) {
        if ($Part -in @('', '.', '..') -or $Part -match '[\x00-\x1f<>:"\\|?*]' -or
            $Part -match '[. ]$' -or $Part -match '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)') {
            throw "Unsafe Windows path: $Path"
        }
    }
}

function Assert-NoReparse {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        $Item = Get-Item -LiteralPath $Path -Force
        if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse points cannot be deployed: $Path"
        }
    }
}

function Resolve-AppFile {
    param([string]$Root, [string]$Path)
    Assert-AppPath $Path
    Assert-NoReparse $Root
    $Resolved = $Root
    foreach ($Part in $Path.Split('/')) {
        $Resolved = Join-Path $Resolved $Part
        Assert-NoReparse $Resolved
    }
    return $Resolved
}

function Get-StateMap {
    param($State)
    Assert-Properties $State @('schema', 'release_id', 'commit', 'files')
    if ($State.schema -ne 1 -or $State.release_id -notmatch '^[A-Za-z0-9._-]+$' -or
        $State.commit -notmatch '^[0-9a-f]{40,64}$' -or $State.files -isnot [Array]) {
        throw 'Invalid deployment state.'
    }
    $Map = New-Object 'Collections.Generic.Dictionary[string,object]' ([StringComparer]::Ordinal)
    $Names = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($Entry in $State.files) {
        Assert-Properties $Entry @('path', 'sha256', 'size')
        Assert-AppPath $Entry.path
        if ($Entry.sha256 -cnotmatch '^[0-9a-f]{64}$' -or
            ($Entry.size -isnot [int] -and $Entry.size -isnot [long]) -or
            $Entry.size -lt 0 -or -not $Names.Add($Entry.path)) {
            throw "Invalid/duplicate file identity: $($Entry.path)"
        }
        $Map.Add($Entry.path, $Entry)
    }
    foreach ($Path in $Map.Keys) {
        $Parts = $Path.Split('/')
        for ($Index = 2; $Index -lt $Parts.Length; $Index++) {
            if ($Names.Contains(($Parts[0..($Index - 1)] -join '/'))) {
                throw "File/directory collision: $Path"
            }
        }
    }
    if (-not $Map.ContainsKey('app/__init__.py')) { throw 'State has no app/__init__.py.' }
    return ,$Map
}

function Get-TextHash {
    param([string]$Text)
    $Hash = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($Hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($Text.Trim())))).Replace('-', '').ToLowerInvariant()
    }
    finally { $Hash.Dispose() }
}

function Assert-FileIdentity {
    param([string]$Path, $Entry)
    Assert-NoReparse $Path
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or
        (Get-Item -LiteralPath $Path).Length -ne $Entry.size -or
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Entry.sha256) {
        throw "File differs from deployment manifest: $Path. Use --full to restore the committed app."
    }
}

function Get-PathSet {
    param($Paths)
    if ($Paths -isnot [Array]) { throw 'Expected a deployment path array.' }
    $Set = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
    foreach ($Path in $Paths) {
        Assert-AppPath $Path
        if (-not $Set.Add($Path)) { throw "Duplicate operation: $Path" }
    }
    return ,$Set
}

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container) -or
    -not (Test-Path -LiteralPath $TargetApp -PathType Container)) {
    throw 'Project root and existing app directory are required.'
}
Assert-NoReparse $ProjectRoot
Assert-NoReparse $TargetApp
Assert-NoReparse $StatePath
Assert-NoReparse $LockPath

# The OS releases this lock on crash. A leftover lock file is harmless.
$DeployLock = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    $OldStateText = $null
    $Baseline = $null
    if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
        $OldStateText = [IO.File]::ReadAllText($StatePath, [Text.Encoding]::UTF8).Trim()
        if ($ReadState) { $null = Get-StateMap ($OldStateText | ConvertFrom-Json) }
    }
    if ($ReadState) {
        if ($null -eq $OldStateText) { Write-Output 'null' }
        else { Write-Output $OldStateText }
        return
    }
    if (-not $ReleaseId -or -not $ArchiveSha256) { throw 'ReleaseId and ArchiveSha256 are required.' }
    $ArchivePath = Resolve-WindowsPath $ArchivePath
    if ((Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $ArchiveSha256) {
        throw 'Deployment archive checksum mismatch.'
    }
    if ($ServiceName -and $RestartBatch) { throw 'Set ServiceName or RestartBatch, not both.' }
    if ($RestartBatch) {
        $RestartBatch = Resolve-WindowsPath $RestartBatch
        if (-not (Test-Path -LiteralPath $RestartBatch -PathType Leaf)) { throw 'Restart batch does not exist.' }
    }
    Assert-NoReparse (Join-Path $ProjectRoot '.nte-deploy')
    if (Test-Path -LiteralPath $WorkRoot) { throw "Release work directory already exists: $WorkRoot" }
    New-Item -ItemType Directory -Path $StageRoot | Out-Null
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Zip = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        $ZipNames = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        foreach ($Entry in $Zip.Entries) {
            $Name = $Entry.FullName
            if (-not $ZipNames.Add($Name)) { throw "Duplicate ZIP entry: $Name" }
            if ($Name -ceq 'manifest.json' -or $Name -ceq 'state.json') { continue }
            if (-not $Name.StartsWith('payload/', [StringComparison]::Ordinal)) { throw "Unexpected ZIP entry: $Name" }
            Assert-AppPath $Name.Substring(8)
        }
    }
    finally { $Zip.Dispose() }
    [IO.Compression.ZipFile]::ExtractToDirectory($ArchivePath, $StageRoot)
    $Manifest = [IO.File]::ReadAllText((Join-Path $StageRoot 'manifest.json')) | ConvertFrom-Json
    $StateText = [IO.File]::ReadAllText((Join-Path $StageRoot 'state.json'), [Text.Encoding]::UTF8).Trim()
    $State = $StateText | ConvertFrom-Json
    $Target = Get-StateMap $State
    Assert-Properties $Manifest @('schema', 'kind', 'base_sha256', 'upserts', 'deletes')
    if ($Manifest.schema -ne 1 -or $Manifest.kind -cnotin @('full', 'delta') -or $State.release_id -cne $ReleaseId) {
        throw 'Package identity mismatch.'
    }
    $Upserts = Get-PathSet $Manifest.upserts
    $Deletes = Get-PathSet $Manifest.deletes
    if ($Manifest.kind -ceq 'delta') {
        if ($null -eq $OldStateText -or (Get-TextHash $OldStateText) -cne $Manifest.base_sha256) {
            throw 'Server deployment base changed. Rebuild the delta package.'
        }
        $Baseline = Get-StateMap ($OldStateText | ConvertFrom-Json)
        # Verify every base file, including files that will be replaced/deleted.
        foreach ($Entry in $Baseline.Values) {
            Assert-FileIdentity (Resolve-AppFile $ProjectRoot $Entry.path) $Entry
        }
    }
    elseif ($null -ne $Manifest.base_sha256 -or $Deletes.Count -ne 0) { throw 'Invalid full package.' }
    $ExpectedUpserts = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
    $ExpectedDeletes = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
    foreach ($Entry in $Target.Values) {
        if ($null -eq $Baseline -or -not $Baseline.ContainsKey($Entry.path) -or
            $Baseline[$Entry.path].sha256 -cne $Entry.sha256 -or $Baseline[$Entry.path].size -ne $Entry.size) {
            $null = $ExpectedUpserts.Add($Entry.path)
        }
    }
    if ($null -ne $Baseline) {
        foreach ($Path in $Baseline.Keys) {
            if (-not $Target.ContainsKey($Path)) { $null = $ExpectedDeletes.Add($Path) }
        }
    }
    if (-not $Upserts.SetEquals($ExpectedUpserts) -or -not $Deletes.SetEquals($ExpectedDeletes) -or
        $ZipNames.Count -ne (2 + $Upserts.Count)) { throw 'Package operations do not match its file inventories.' }
    foreach ($Path in $Upserts) {
        if (-not $ZipNames.Contains('payload/' + $Path)) { throw "Missing payload: $Path" }
        Assert-FileIdentity (Resolve-AppFile (Join-Path $StageRoot 'payload') $Path) $Target[$Path]
    }

    # Build a complete next app from unchanged baseline files and the delta.
    # This excludes caches/untracked runtime files and also supports case-only renames
    # and file-to-directory or directory-to-file replacements.
    New-Item -ItemType Directory -Path $IncomingApp | Out-Null
    if ($null -ne $Baseline) {
        foreach ($Path in $Target.Keys) {
            if ($Upserts.Contains($Path)) { continue }
            $Source = Resolve-AppFile $ProjectRoot $Path
            $Destination = Resolve-AppFile $IncomingRoot $Path
            $null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination)
            Copy-Item -LiteralPath $Source -Destination $Destination
        }
    }
    foreach ($Path in $Upserts) {
        $Source = Resolve-AppFile (Join-Path $StageRoot 'payload') $Path
        $Destination = Resolve-AppFile $IncomingRoot $Path
        $null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination)
        Copy-Item -LiteralPath $Source -Destination $Destination
    }
    foreach ($Entry in $Target.Values) {
        Assert-FileIdentity (Resolve-AppFile $IncomingRoot $Entry.path) $Entry
    }
    $NewStatePath = Join-Path $WorkRoot 'new-state.json'
    [IO.File]::WriteAllText($NewStatePath, $StateText, (New-Object Text.UTF8Encoding($false)))
    if ($null -ne $OldStateText) {
        [IO.File]::WriteAllText((Join-Path $WorkRoot 'backup-state.json'), $OldStateText, (New-Object Text.UTF8Encoding($false)))
    }
    $OldAppMoved = $false
    $Activated = $false
    $StoppedProcessCount = 0
    $ServiceStopped = $false
    $NoChanges = $Manifest.kind -ceq 'delta' -and $Upserts.Count -eq 0 -and $Deletes.Count -eq 0
    try {
        if (-not $NoChanges) {
            if ($ServiceName) {
                $ServiceStopped = $true
                Stop-Service -Name $ServiceName -ErrorAction Stop
                (Get-Service -Name $ServiceName).WaitForStatus([ServiceProcess.ServiceControllerStatus]::Stopped, [TimeSpan]::FromSeconds(30))
            }
            elseif ($RestartBatch) { $StoppedProcessCount = Stop-PortListener $ListenPort }
            Move-Item -LiteralPath $TargetApp -Destination $BackupApp
            $OldAppMoved = $true
            Move-Item -LiteralPath $IncomingApp -Destination $TargetApp
            $Activated = $true
            if ($ServiceName) {
                Start-Service -Name $ServiceName -ErrorAction Stop
                (Get-Service -Name $ServiceName).WaitForStatus([ServiceProcess.ServiceControllerStatus]::Running, [TimeSpan]::FromSeconds(30))
            }
            elseif ($RestartBatch) {
                Start-WaitressBatch $RestartBatch $ProjectRoot
                if (-not (Wait-ForPort $ListenPort 30)) { throw 'Waitress startup verification failed.' }
            }
        }
        # Publish the new baseline only after successful activation/health checks.
        if (Test-Path -LiteralPath $StatePath) {
            [IO.File]::Replace($NewStatePath, $StatePath, (Join-Path $WorkRoot 'replaced-state.json'))
        }
        else { [IO.File]::Move($NewStatePath, $StatePath) }
    }
    catch {
        $DeploymentError = $_
        try {
            if ($Activated -and $ServiceName) { Stop-Service -Name $ServiceName -ErrorAction Stop }
            elseif ($Activated -and $RestartBatch) { $null = Stop-PortListener $ListenPort }
        }
        catch { Write-Warning 'Failed to stop the new app; attempting directory rollback.' }
        if ($OldAppMoved) {
            if (Test-Path -LiteralPath $TargetApp) { Move-Item -LiteralPath $TargetApp -Destination $FailedApp }
            Move-Item -LiteralPath $BackupApp -Destination $TargetApp
        }
        if ($ServiceStopped) { Start-Service -Name $ServiceName -ErrorAction SilentlyContinue }
        elseif ($RestartBatch -and $StoppedProcessCount -gt 0) {
            try { Start-WaitressBatch $RestartBatch $ProjectRoot }
            catch { Write-Warning 'Previous app restored; Waitress restart failed.' }
        }
        throw $DeploymentError
    }
    # Cleanup failures must not turn a committed deployment into a reported rollback.
    $CleanupPaths = @($StageRoot, $IncomingRoot, $ArchivePath)
    $UploadedHelper = Join-Path (Split-Path -Parent $ArchivePath) ("deploy_windows_app_" + $ReleaseId + '.ps1')
    if ($PSCommandPath -eq $UploadedHelper) { $CleanupPaths += $PSCommandPath }
    if ($NoChanges) { $CleanupPaths += $WorkRoot }
    foreach ($Path in $CleanupPaths) {
        if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Continue }
    }
    Write-Output "DEPLOY_SUCCESS release=$ReleaseId kind=$($Manifest.kind) upserts=$($Upserts.Count) deletes=$($Deletes.Count) no_changes=$NoChanges"
    Write-Output "APP_PATH=$TargetApp"
    Write-Output "STATE_PATH=$StatePath"
    if ($OldAppMoved) { Write-Output "BACKUP_PATH=$BackupApp" }
}
finally { $DeployLock.Dispose() }
