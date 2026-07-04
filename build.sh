#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SDK_ENV="${SDK_ENV:-}"
BUILD_DIR="${BUILD_DIR:-$APP_ROOT/build}"
HOST_BUILD_DIR="${HOST_BUILD_DIR:-$APP_ROOT/build-host}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

BOARD_USER="${BOARD_USER:-petalinux}"
BOARD_HOST="${BOARD_HOST:-192.168.1.4}"
BOARD_DIR="${BOARD_DIR:-/home/petalinux}"
RUN_PREFIX="${RUN_PREFIX:-sudo}"

TARGET_APPS=(DmaApp AccUnitTest)

# shellcheck disable=SC1091
source "$APP_ROOT/scripts/cross_env.sh"

usage() {
    cat <<'USAGE'
Usage:
  ./build.sh [command] [args...]

Commands:
  build              Cross-compile DmaApp and AccUnitTest. Default command.
  deploy             Build, then copy target binaries to the board.
  run [app] [args]   Build, deploy, then run app on the board.
                     If app is omitted or starts with "-", DmaApp is used.
  remote [app] [args] Run an already deployed app on the board.
  host               Build the host-only YoloCppCheck utility.
  host-run           Build and run YoloCppCheck locally.
  clean              Remove target and host build directories.
  help               Show this help.

Environment overrides:
  SDK_ENV=/path/to/environment-setup-...
  BOARD_USER=petalinux BOARD_HOST=192.168.1.4 BOARD_DIR=/home/petalinux
  RUN_PREFIX=sudo     Set RUN_PREFIX= to run without sudo.
  BUILD_DIR=... HOST_BUILD_DIR=... JOBS=...

Examples:
  ./build.sh build
  ./build.sh deploy
  ./build.sh run --basic --verify-all
  ./build.sh run DmaApp --scan-overlap --tasks 32 --timeout-ms 10000
  RUN_PREFIX= ./build.sh run AccUnitTest
USAGE
}

build_target() {
    load_cross_env
    echo "Using cross environment: $CROSS_ENV_SOURCE"
    reset_stale_cmake_cache "$BUILD_DIR"
    cmake -S "$APP_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$APP_ROOT/toolchain.cmake" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    cmake --build "$BUILD_DIR" -j "$JOBS"
}

build_host() {
    reset_stale_cmake_cache "$HOST_BUILD_DIR"
    cmake -S "$APP_ROOT" -B "$HOST_BUILD_DIR" \
        -DBUILD_HOST_YOLO_CPP=ON \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    cmake --build "$HOST_BUILD_DIR" -j "$JOBS"
}

reset_stale_cmake_cache() {
    local dir="$1"
    local cache="$dir/CMakeCache.txt"
    local cached_source=""
    local cached_compiler=""
    local cached_sysroot=""
    local cached_c_flags=""
    local cached_cxx_flags=""
    local cached_linker_flags=""
    local current_compiler=""

    if [[ ! -f "$cache" ]]; then
        return
    fi

    cached_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache" | tail -1)"
    if [[ -n "$cached_source" && "$cached_source" != "$APP_ROOT" ]]; then
        echo "Removing stale CMake build directory: $dir" >&2
        echo "  cached source: $cached_source" >&2
        echo "  current source: $APP_ROOT" >&2
        rm -rf "$dir"
        return
    fi

    if [[ -n "${CC:-}" ]]; then
        current_compiler="$(command -v "${CC%% *}" 2>/dev/null || true)"
    fi
    cached_compiler="$(sed -n 's/^CMAKE_C_COMPILER:FILEPATH=//p' "$cache" | tail -1)"
    cached_sysroot="$(sed -n 's/^CMAKE_SYSROOT:PATH=//p' "$cache" | tail -1)"
    cached_c_flags="$(sed -n 's/^CMAKE_C_FLAGS:STRING=//p' "$cache" | tail -1)"
    cached_cxx_flags="$(sed -n 's/^CMAKE_CXX_FLAGS:STRING=//p' "$cache" | tail -1)"
    cached_linker_flags="$(sed -n 's/^CMAKE_EXE_LINKER_FLAGS:STRING=//p' "$cache" | tail -1)"
    if [[ -n "$current_compiler" && -n "$cached_compiler" && "$cached_compiler" != "$current_compiler" ]]; then
        echo "Removing CMake build directory with stale compiler: $dir" >&2
        echo "  cached compiler: $cached_compiler" >&2
        echo "  current compiler: $current_compiler" >&2
        rm -rf "$dir"
        return
    fi
    if [[ -n "${SDKTARGETSYSROOT:-}" && -n "$cached_sysroot" && "$cached_sysroot" != "$SDKTARGETSYSROOT" ]]; then
        echo "Removing CMake build directory with stale sysroot: $dir" >&2
        echo "  cached sysroot: $cached_sysroot" >&2
        echo "  current sysroot: $SDKTARGETSYSROOT" >&2
        rm -rf "$dir"
        return
    fi
    if [[ "$cached_c_flags $cached_cxx_flags $cached_linker_flags" == *"/tmp/sysroots/"* ]]; then
        echo "Removing CMake build directory with stale PetaLinux flags: $dir" >&2
        rm -rf "$dir"
    fi
}

deploy_target() {
    build_target
    for app in "${TARGET_APPS[@]}"; do
        if [[ ! -x "$BUILD_DIR/$app" ]]; then
            echo "missing built target: $BUILD_DIR/$app" >&2
            exit 1
        fi
        scp "$BUILD_DIR/$app" "$BOARD_USER@$BOARD_HOST:$BOARD_DIR/"
    done
    ssh "$BOARD_USER@$BOARD_HOST" "cd $(printf '%q' "$BOARD_DIR") && chmod +x ${TARGET_APPS[*]}"
}

remote_quote_args() {
    local out=()
    local arg
    for arg in "$@"; do
        out+=("$(printf '%q' "$arg")")
    done
    printf '%s ' "${out[@]}"
}

run_remote() {
    local app="DmaApp"
    if [[ $# -gt 0 && "$1" != -* ]]; then
        app="$1"
        shift
    fi

    local run_prefix=""
    if [[ -n "$RUN_PREFIX" ]]; then
        run_prefix="$(printf '%q ' "$RUN_PREFIX")"
    fi

    local cmd
    cmd="cd $(printf '%q' "$BOARD_DIR") && ${run_prefix}./$(printf '%q' "$app")"
    if [[ $# -gt 0 ]]; then
        cmd+=" $(remote_quote_args "$@")"
    fi

    ssh "$BOARD_USER@$BOARD_HOST" "$cmd"
}

clean() {
    rm -rf "$BUILD_DIR" "$HOST_BUILD_DIR"
}

cmd="${1:-build}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$cmd" in
    build)
        build_target
        ;;
    deploy)
        deploy_target
        ;;
    run)
        deploy_target
        run_remote "$@"
        ;;
    remote)
        run_remote "$@"
        ;;
    host)
        build_host
        ;;
    host-run)
        build_host
        "$HOST_BUILD_DIR/YoloCppCheck" "$@"
        ;;
    clean)
        clean
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "unknown command: $cmd" >&2
        usage >&2
        exit 2
        ;;
esac
