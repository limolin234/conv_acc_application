#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ENV="${SDK_ENV:-}"

# shellcheck disable=SC1091
source "$APP_ROOT/scripts/cross_env.sh"

if ! load_cross_env; then
    return 2 2>/dev/null || exit 2
fi

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ $# -eq 0 ]]; then
        echo "PetaLinux SDK environment loaded for this subprocess."
        echo "Using cross environment: $CROSS_ENV_SOURCE"
        echo "For an interactive shell, run: source ./env.sh"
    else
        exec "$@"
    fi
fi
