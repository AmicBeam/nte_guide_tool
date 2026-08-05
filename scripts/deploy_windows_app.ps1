[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$ArchivePath,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$ReleaseId,

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
$ArchivePath = Resolve-WindowsPath -Path $ArchivePath
$TargetApp = Join-Path $ProjectRoot 'app'
$WorkRoot = Join-Path $env:TEMP ("nte_board_game_app_" + $ReleaseId)
$StageRoot = Join-Path $WorkRoot 'stage'
$IncomingApp = Join-Path $StageRoot 'app'
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

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Project root does not exist: $ProjectRoot"
}
if (-not (Test-Path -LiteralPath $TargetApp -PathType Container)) {
    throw "Target app directory does not exist: $TargetApp"
}
if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
    throw "Deployment archive does not exist: $ArchivePath"
}
if (Test-Path -LiteralPath $WorkRoot) {
    throw "Release work directory already exists: $WorkRoot"
}
if ($ServiceName -and $RestartBatch) {
    throw 'Set either ServiceName or RestartBatch, not both.'
}
if ($RestartBatch) {
    $RestartBatch = Resolve-WindowsPath -Path $RestartBatch
    if (-not (Test-Path -LiteralPath $RestartBatch -PathType Leaf)) {
        throw "Restart batch file does not exist: $RestartBatch"
    }
}

New-Item -ItemType Directory -Path $StageRoot | Out-Null

Add-Type -AssemblyName System.IO.Compression.FileSystem

$Zip = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
try {
    foreach ($Entry in $Zip.Entries) {
        $EntryPath = $Entry.FullName.Replace('\', '/')
        $PathSegments = @($EntryPath.Split('/') | Where-Object { $_ -ne '' })
        if (
            $EntryPath.StartsWith('/') -or
            $EntryPath -match '^[A-Za-z]:' -or
            $PathSegments -contains '..'
        ) {
            throw "Archive contains an unsafe path: $($Entry.FullName)"
        }
    }
}
finally {
    $Zip.Dispose()
}

[System.IO.Compression.ZipFile]::ExtractToDirectory($ArchivePath, $StageRoot)

if (-not (Test-Path -LiteralPath (Join-Path $IncomingApp '__init__.py') -PathType Leaf)) {
    throw 'Archive does not contain a complete app package.'
}

$OldAppMoved = $false
$NewAppActivated = $false
$StoppedProcessCount = 0

try {
    if ($RestartBatch) {
        $StoppedProcessCount = Stop-PortListener -Port $ListenPort
    }

    Move-Item -LiteralPath $TargetApp -Destination $BackupApp
    $OldAppMoved = $true

    Move-Item -LiteralPath $IncomingApp -Destination $TargetApp
    $NewAppActivated = $true

    if ($ServiceName) {
        Restart-Service -Name $ServiceName -ErrorAction Stop
        $Service = Get-Service -Name $ServiceName
        $Service.WaitForStatus(
            [System.ServiceProcess.ServiceControllerStatus]::Running,
            [TimeSpan]::FromSeconds(30)
        )
    }
    elseif ($RestartBatch) {
        Start-WaitressBatch -BatchPath $RestartBatch -WorkingDirectory $ProjectRoot
        if (-not (Wait-ForPort -Port $ListenPort -TimeoutSeconds 30)) {
            throw "Waitress did not begin listening on port $ListenPort within 30 seconds."
        }
    }
}
catch {
    $DeploymentError = $_

    if ($NewAppActivated -and (Test-Path -LiteralPath $TargetApp)) {
        Move-Item -LiteralPath $TargetApp -Destination $FailedApp
    }
    if ($OldAppMoved -and (Test-Path -LiteralPath $BackupApp)) {
        Move-Item -LiteralPath $BackupApp -Destination $TargetApp
    }
    if ($ServiceName) {
        try {
            Start-Service -Name $ServiceName -ErrorAction SilentlyContinue
        }
        catch {
            Write-Warning "The previous app was restored, but service restart also failed."
        }
    }
    elseif ($RestartBatch -and $StoppedProcessCount -gt 0) {
        try {
            Start-WaitressBatch -BatchPath $RestartBatch -WorkingDirectory $ProjectRoot
        }
        catch {
            Write-Warning "The previous app was restored, but Waitress restart also failed."
        }
    }

    throw $DeploymentError
}

if (Test-Path -LiteralPath $StageRoot) {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force
}
if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

Write-Output "DEPLOY_SUCCESS release=$ReleaseId"
Write-Output "APP_PATH=$TargetApp"
Write-Output "BACKUP_PATH=$BackupApp"
if (-not $ServiceName) {
    if ($RestartBatch) {
        Write-Output "RESTARTED_BATCH=$RestartBatch"
        Write-Output "LISTEN_PORT=$ListenPort"
        Write-Output "WAITRESS_LOG=$(Join-Path $ProjectRoot 'logs\waitress-deploy.log')"
    }
    else {
        Write-Warning 'No Flask process was restarted.'
    }
}
