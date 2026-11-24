#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_APP="$REPO_ROOT/app"
OUTPUT_DIR="$REPO_ROOT/dist/deploy"
REMOTE_SCRIPT="$SCRIPT_DIR/deploy_windows_app.ps1"
PACKAGE_SCRIPT="$SCRIPT_DIR/deploy_app_package.py"
LOCAL_CONFIG="$SCRIPT_DIR/deploy_windows_app.local.env"

if [[ -f "$LOCAL_CONFIG" ]]; then
    # shellcheck disable=SC1090
    source "$LOCAL_CONFIG"
fi

DEPLOY_HOST="${NTE_DEPLOY_HOST:-}"
DEPLOY_PROJECT="${NTE_DEPLOY_PROJECT:-}"
DEPLOY_SERVICE="${NTE_DEPLOY_SERVICE:-}"
DEPLOY_RESTART_BATCH="${NTE_DEPLOY_RESTART_BATCH-start_waitress.bat}"
DEPLOY_LISTEN_PORT="${NTE_DEPLOY_LISTEN_PORT:-8000}"
REMOTE_TEMP="C:/Windows/Temp"

MODE="check"
ALLOW_DIRTY_APP="false"
FORCE_FULL="false"
BASE_STATE=""
TEMP_BASE_STATE=""
SSH_CONTROL_DIR=""
SSH_MASTER_STARTED="false"
SSH_OPTIONS=()

cleanup() {
    local result=$?
    if [[ "$SSH_MASTER_STARTED" == "true" ]]; then
        ssh "${SSH_OPTIONS[@]}" -O exit "$DEPLOY_HOST" >/dev/null 2>&1 || true
    fi
    [[ -z "$TEMP_BASE_STATE" ]] || rm -f "$TEMP_BASE_STATE" || true
    [[ -z "$SSH_CONTROL_DIR" ]] || rm -rf "$SSH_CONTROL_DIR" || true
    return "$result"
}
trap cleanup EXIT

usage() {
    cat <<'EOF'
Usage:
  scripts/deploy_windows_app.sh --check [--allow-dirty-app]
  scripts/deploy_windows_app.sh --prepare [--allow-dirty-app]
  scripts/deploy_windows_app.sh --deploy [--allow-dirty-app]

Modes:
  --check    Validate the committed app source and print the deployment target.
  --prepare  Build a full ZIP, or a delta with --base-state. No server contact.
  --deploy   Read server state, upload changed files/deletions, and activate.

Options:
  --full    Force a complete package (first deployment does this automatically).
  --base-state PATH
            Offline --prepare only: compare with a saved server state JSON.
  --allow-dirty-app
              Continue when app/ has local changes. The package still contains
              only the committed app/ tree from Git HEAD.

Environment overrides:
  NTE_DEPLOY_HOST       SSH target (required)
  NTE_DEPLOY_PROJECT    Windows project path (required)
  NTE_DEPLOY_SERVICE    Optional Windows service to restart
  NTE_DEPLOY_RESTART_BATCH
                        Batch file used when no service is set
                        (default: start_waitress.bat; set empty to disable)
  NTE_DEPLOY_LISTEN_PORT
                        Port used to stop and verify Waitress (default: 8000)

By default, the script refuses to check, prepare, or deploy while app/ has
staged, unstaged, or untracked changes. It never uploads files outside the
committed app/ tree.
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

# Pass Windows paths as PowerShell string literals, never interpolated shell code.
encode_remote_command() {
    python3 - "$@" <<'PY'
import base64, sys
def quote(value):
    return "'" + value.replace("'", "''") + "'"
parts = ["$ProgressPreference = 'SilentlyContinue';", '&', quote(sys.argv[1])]
args = sys.argv[2:]
allowed = {'-ProjectRoot', '-ArchivePath', '-ArchiveSha256', '-ReleaseId',
           '-ServiceName', '-RestartBatch', '-ListenPort'}
while args:
    flag = args.pop(0)
    if flag == '-ReadState':
        parts.append(flag)
    else:
        if flag not in allowed or not args:
            raise ValueError('Invalid remote parameter')
        parts.extend((flag, quote(args.pop(0))))
print(base64.b64encode(' '.join(parts).encode('utf-16le')).decode('ascii'))
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            MODE="check"
            ;;
        --prepare)
            MODE="prepare"
            ;;
        --deploy)
            MODE="deploy"
            ;;
        --allow-dirty-app)
            ALLOW_DIRTY_APP="true"
            ;;
        --full)
            FORCE_FULL="true"
            ;;
        --base-state)
            [[ $# -ge 2 ]] || fail "--base-state requires a JSON path"
            BASE_STATE="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
    shift
done

[[ -d "$LOCAL_APP" ]] || fail "local app directory not found: $LOCAL_APP"
[[ -f "$LOCAL_APP/__init__.py" ]] || fail "local app package is incomplete"
[[ -f "$REMOTE_SCRIPT" ]] || fail "server deployment script not found: $REMOTE_SCRIPT"
[[ -f "$PACKAGE_SCRIPT" ]] || fail "package builder not found: $PACKAGE_SCRIPT"
[[ -z "$BASE_STATE" || "$MODE" == "prepare" ]] || fail "--base-state is only supported with --prepare"
[[ -z "$BASE_STATE" || "$FORCE_FULL" != "true" ]] || fail "--full and --base-state cannot be combined"
[[ -n "$DEPLOY_HOST" ]] || fail "set NTE_DEPLOY_HOST or create $LOCAL_CONFIG"
[[ -n "$DEPLOY_PROJECT" ]] || fail "set NTE_DEPLOY_PROJECT or create $LOCAL_CONFIG"
[[ "$DEPLOY_LISTEN_PORT" =~ ^[0-9]+$ ]] || fail "NTE_DEPLOY_LISTEN_PORT must be numeric"
(( DEPLOY_LISTEN_PORT >= 1 && DEPLOY_LISTEN_PORT <= 65535 )) || fail "invalid listen port"

require_command git
require_command shasum
require_command python3

if [[ "$MODE" == "deploy" ]]; then
    require_command ssh
    require_command scp
fi

git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
    fail "repository metadata is unavailable"

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
GIT_REV="$(git -C "$REPO_ROOT" rev-parse --short "$GIT_COMMIT")"
GIT_BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
APP_CHANGES="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all -- app)"
FILE_COUNT="$(git -C "$REPO_ROOT" ls-tree -r --name-only "$GIT_COMMIT" app | wc -l | tr -d ' ')"

printf 'Source: %s@%s (%s files)\n' "${GIT_BRANCH:-detached}" "$GIT_REV" "$FILE_COUNT"
printf 'Target: %s — %s/app\n' "$DEPLOY_HOST" "$DEPLOY_PROJECT"
if [[ -n "$DEPLOY_SERVICE" ]]; then
    printf 'Restart service: %s\n' "$DEPLOY_SERVICE"
elif [[ -n "$DEPLOY_RESTART_BATCH" ]]; then
    printf 'Restart batch:   %s (port %s)\n' "$DEPLOY_RESTART_BATCH" "$DEPLOY_LISTEN_PORT"
else
    printf 'Process restart: disabled\n'
fi

if [[ -n "$APP_CHANGES" ]]; then
    printf 'Uncommitted app changes detected:\n' >&2
    printf '%s\n' "$APP_CHANGES" >&2
    if [[ "$ALLOW_DIRTY_APP" != "true" ]]; then
        fail "commit or discard all app/ changes, or pass --allow-dirty-app to package Git HEAD only"
    fi
    printf 'WARNING: continuing with --allow-dirty-app; local app/ changes are excluded from the package.\n' >&2
fi

if [[ "$MODE" == "check" ]]; then
    if [[ -n "$APP_CHANGES" ]]; then
        printf 'Check complete. Package source is Git HEAD; local app/ changes will be excluded. No network connection was made.\n'
    else
        printf 'Check complete. app/ matches Git HEAD; no network connection was made.\n'
    fi
    exit 0
fi

RELEASE_ID="$(date -u '+%Y%m%dT%H%M%SZ')-$GIT_REV-$(python3 -c 'import secrets; print(secrets.token_hex(4))')"

mkdir -p "$OUTPUT_DIR"
ARCHIVE_NAME="nte_board_game_app_${RELEASE_ID}.zip"
ARCHIVE_PATH="$OUTPUT_DIR/$ARCHIVE_NAME"

REMOTE_HELPER="$REMOTE_TEMP/deploy_windows_app_${RELEASE_ID}.ps1"
if [[ "$MODE" == "deploy" ]]; then
    # A short private socket path avoids macOS Unix socket length limits.
    # Authenticate once; later channels fail rather than prompting again if lost.
    SSH_CONTROL_DIR="$(mktemp -d /tmp/nte-deploy-ssh.XXXXXX)"
    chmod 700 "$SSH_CONTROL_DIR"
    SSH_OPTIONS=(-o "ControlPath=$SSH_CONTROL_DIR/socket" -o ControlMaster=no -o BatchMode=yes -o ProxyCommand=false
                 -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
    printf 'Connecting (15-second SSH handshake timeout; one authentication for this deployment)...\n'
    if ! ssh -o "ControlPath=$SSH_CONTROL_DIR/socket" -o ControlMaster=yes -o ControlPersist=no \
        -o ConnectTimeout=15 -o ConnectionAttempts=1 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -f -N "$DEPLOY_HOST"; then
        fail "Could not establish the deployment SSH session. Nothing was uploaded."
    fi
    SSH_MASTER_STARTED="true"
    scp -q "${SSH_OPTIONS[@]}" "$REMOTE_SCRIPT" "$DEPLOY_HOST:$REMOTE_HELPER"
    if [[ "$FORCE_FULL" != "true" ]]; then
        TEMP_BASE_STATE="$(mktemp "${TMPDIR:-/tmp}/nte-deploy-base.XXXXXX")"
        READ_COMMAND="$(encode_remote_command "$REMOTE_HELPER" -ProjectRoot "$DEPLOY_PROJECT" -ReadState)"
        printf 'Reading server deployment state...\n'
        ssh "${SSH_OPTIONS[@]}" "$DEPLOY_HOST" "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $READ_COMMAND" > "$TEMP_BASE_STATE"
        BASE_STATE="$TEMP_BASE_STATE"
    fi
fi

PACKAGE_COMMAND=(python3 "$PACKAGE_SCRIPT" --repo "$REPO_ROOT" --revision "$GIT_COMMIT" --output "$ARCHIVE_PATH" --release-id "$RELEASE_ID")
if [[ -n "$BASE_STATE" ]]; then
    PACKAGE_COMMAND+=(--base-state "$BASE_STATE")
fi
"${PACKAGE_COMMAND[@]}"

printf 'Prepared archive: %s\n' "$ARCHIVE_PATH"
ARCHIVE_SHA256="$(shasum -a 256 "$ARCHIVE_PATH" | awk '{print $1}')"

if [[ "$MODE" == "prepare" ]]; then
    printf 'Preparation complete. Nothing was uploaded.\n'
    exit 0
fi

REMOTE_ARCHIVE="$REMOTE_TEMP/$ARCHIVE_NAME"

printf 'Uploading app package...\n'
scp "${SSH_OPTIONS[@]}" "$ARCHIVE_PATH" "$DEPLOY_HOST:$REMOTE_TEMP/"

REMOTE_COMMAND=(
    "$REMOTE_HELPER"
    -ProjectRoot "$DEPLOY_PROJECT"
    -ArchivePath "$REMOTE_ARCHIVE"
    -ArchiveSha256 "$ARCHIVE_SHA256"
    -ReleaseId "$RELEASE_ID"
)

if [[ -n "$DEPLOY_SERVICE" ]]; then
    REMOTE_COMMAND+=(-ServiceName "$DEPLOY_SERVICE")
elif [[ -n "$DEPLOY_RESTART_BATCH" ]]; then
    REMOTE_COMMAND+=(
        -RestartBatch "$DEPLOY_PROJECT/$DEPLOY_RESTART_BATCH"
        -ListenPort "$DEPLOY_LISTEN_PORT"
    )
fi

printf 'Activating release %s...\n' "$RELEASE_ID"
ACTIVATE_COMMAND="$(encode_remote_command "${REMOTE_COMMAND[@]}")"
ssh "${SSH_OPTIONS[@]}" "$DEPLOY_HOST" "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $ACTIVATE_COMMAND"
if ! rm -f "$ARCHIVE_PATH"; then
    printf 'WARNING: deployment succeeded; failed to clean local archive %s\n' "$ARCHIVE_PATH" >&2
fi
printf 'Deployment complete. Managed target: %s/app\n' "$DEPLOY_PROJECT"
