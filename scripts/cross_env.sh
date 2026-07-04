#!/usr/bin/env bash

if [[ -z "${APP_ROOT:-}" ]]; then
    APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

SDK_ENV="${SDK_ENV:-}"

compiler_available() {
    local cc_cmd="${CC:-}"
    cc_cmd="${cc_cmd%% *}"
    [[ -n "$cc_cmd" ]] && command -v "$cc_cmd" >/dev/null 2>&1
}

setup_vitis_aarch32_gnu() {
    local root
    local candidates=(
        "$APP_ROOT/toolchains/vitis-aarch32-gcc"
        "/opt/Xilinx/2025.2/gnu/aarch32/lin/gcc-arm-linux-gnueabi"
    )

    for root in "${candidates[@]}"; do
        if [[ -x "$root/bin/arm-linux-gnueabihf-gcc" && -d "$root/cortexa9t2hf-neon-amd-linux-gnueabi" ]]; then
            unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
            export SDKTARGETSYSROOT="$root/cortexa9t2hf-neon-amd-linux-gnueabi"
            export PATH="$root/bin:$PATH"
            export CC="arm-linux-gnueabihf-gcc -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard --sysroot=$SDKTARGETSYSROOT"
            export CXX="arm-linux-gnueabihf-g++ -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard --sysroot=$SDKTARGETSYSROOT"
            export AR="arm-linux-gnueabihf-ar"
            export AS="arm-linux-gnueabihf-as"
            export LD="arm-linux-gnueabihf-ld"
            export NM="arm-linux-gnueabihf-nm"
            export RANLIB="arm-linux-gnueabihf-ranlib"
            export STRIP="arm-linux-gnueabihf-strip"
            export PKG_CONFIG_SYSROOT_DIR="$SDKTARGETSYSROOT"
            export PKG_CONFIG_PATH="$SDKTARGETSYSROOT/usr/lib/pkgconfig:$SDKTARGETSYSROOT/usr/share/pkgconfig"
            export CROSS_ENV_SOURCE="$root"
            return 0
        fi
    done

    return 1
}

source_petalinux_sdk() {
    local candidate
    local candidates=(
        "$APP_ROOT/toolchains/yocto/environment-setup-cortexa9t2hf-neon-xilinx-linux-gnueabi"
        "$APP_ROOT/../external/petalinux/peta_prj/ax_peta/components/yocto/environment-setup-cortexa9t2hf-neon-xilinx-linux-gnueabi"
        "$HOME/petalinux_project/zynq7020_petalinux/images/linux/sdk/environment-setup-cortexa9t2hf-neon-xilinx-linux-gnueabi"
    )

    if [[ -n "$SDK_ENV" ]]; then
        candidates=("$SDK_ENV")
    fi

    for candidate in "${candidates[@]}"; do
        if [[ ! -f "$candidate" ]]; then
            continue
        fi

        # shellcheck disable=SC1090
        source "$candidate"
        if compiler_available; then
            export CROSS_ENV_SOURCE="$candidate"
            return 0
        fi
    done

    if [[ -n "$SDK_ENV" ]]; then
        echo "SDK_ENV was found but did not expose a usable compiler: $SDK_ENV" >&2
        return 1
    fi

    return 1
}

load_cross_env() {
    if [[ -n "$SDK_ENV" ]]; then
        source_petalinux_sdk || return 1
        return 0
    fi

    if setup_vitis_aarch32_gnu; then
        return 0
    fi

    if source_petalinux_sdk; then
        return 0
    fi

    echo "No usable Zynq aarch32 cross toolchain was found." >&2
    echo "Expected project-local toolchain: $APP_ROOT/toolchains/vitis-aarch32-gcc" >&2
    echo "Or set SDK_ENV=/path/to/environment-setup-cortexa9t2hf-neon-xilinx-linux-gnueabi" >&2
    return 2
}
