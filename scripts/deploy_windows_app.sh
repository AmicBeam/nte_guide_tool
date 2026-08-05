#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_APP="$REPO_ROOT/app"
OUTPUT_DIR="$REPO_ROOT/dist/deploy"
REMOTE_SCRIPT="$SCRIPT_DIR/deploy_windows_app.ps1"
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

usage() {
    cat <<'EOF'
Usage:
  scripts/deploy_windows_app.sh --check
  scripts/deploy_windows_app.sh --prepare
  scripts/deploy_windows_app.sh --deploy

Modes:
  --check    Validate the committed app source and print the deployment target.
  --prepare  Build an app-only ZIP from Git HEAD. Does not contact the server.
  --deploy   Build from Git HEAD, upload, and replace the server app directory.

Environment overrides:
  NTE_DEPLOY_HOST       SSH target (required)
  NTE_DEPLOY_PROJECT    Windows project path (required)
  NTE_DEPLOY_SERVICE    Optional Windows service to restart
  NTE_DEPLOY_RESTART_BATCH
                        Batch file used when no service is set
                        (default: start_waitress.bat; set empty to disable)
  NTE_DEPLOY_LISTEN_PORT
                        Port used to stop and verify Waitress (default: 8000)

The script refuses to prepare or deploy while app/ has staged, unstaged, or
untracked changes. It never uploads files outside the committed app/ tree.
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
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
[[ -n "$DEPLOY_HOST" ]] || fail "set NTE_DEPLOY_HOST or create $LOCAL_CONFIG"
[[ -n "$DEPLOY_PROJECT" ]] || fail "set NTE_DEPLOY_PROJECT or create $LOCAL_CONFIG"
[[ "$DEPLOY_LISTEN_PORT" =~ ^[0-9]+$ ]] || fail "NTE_DEPLOY_LISTEN_PORT must be numeric"
(( DEPLOY_LISTEN_PORT >= 1 && DEPLOY_LISTEN_PORT <= 65535 )) || fail "invalid listen port"

require_command git
require_command shasum

if [[ "$MODE" == "deploy" ]]; then
    require_command ssh
    require_command scp
fi

git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
    fail "repository metadata is unavailable"

GIT_REV="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
GIT_BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
APP_CHANGES="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all -- app)"
FILE_COUNT="$(git -C "$REPO_ROOT" ls-tree -r --name-only HEAD app | wc -l | tr -d ' ')"

printf 'Git source:     %s@%s\n' "${GIT_BRANCH:-detached}" "$GIT_REV"
printf 'Committed files:%s\n' " $FILE_COUNT"
printf 'Remote host:    %s\n' "$DEPLOY_HOST"
printf 'Remote target:  %s/app\n' "$DEPLOY_PROJECT"
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
    fail "commit or discard all app/ changes before preparing or deploying"
fi

if [[ "$MODE" == "check" ]]; then
    printf 'Check complete. app/ matches Git HEAD; no network connection was made.\n'
    exit 0
fi

RELEASE_ID="$(date -u '+%Y%m%dT%H%M%SZ')-$GIT_REV"

mkdir -p "$OUTPUT_DIR"
ARCHIVE_NAME="nte_board_game_app_${RELEASE_ID}.zip"
ARCHIVE_PATH="$OUTPUT_DIR/$ARCHIVE_NAME"

rm -f "$ARCHIVE_PATH"
git -C "$REPO_ROOT" archive --format=zip --output="$ARCHIVE_PATH" HEAD app

printf 'Prepared archive: %s\n' "$ARCHIVE_PATH"
printf 'SHA-256: '
shasum -a 256 "$ARCHIVE_PATH" | awk '{print $1}'
du -h "$ARCHIVE_PATH" | awk '{printf "Archive size: %s\n", $1}'

if [[ "$MODE" == "prepare" ]]; then
    printf 'Preparation complete. Nothing was uploaded.\n'
    exit 0
fi

REMOTE_ARCHIVE="$REMOTE_TEMP/$ARCHIVE_NAME"

printf 'Uploading app package and deployment helper...\n'
scp "$ARCHIVE_PATH" "$REMOTE_SCRIPT" "$DEPLOY_HOST:$REMOTE_TEMP/"

REMOTE_COMMAND=(
    powershell.exe
    -NoProfile
    -ExecutionPolicy Bypass
    -File 'C:\Windows\Temp\deploy_windows_app.ps1'
    -ProjectRoot "$DEPLOY_PROJECT"
    -ArchivePath "$REMOTE_ARCHIVE"
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
ssh "$DEPLOY_HOST" "${REMOTE_COMMAND[*]}"
printf 'Deployment complete. Only %s/app was replaced.\n' "$DEPLOY_PROJECT"
