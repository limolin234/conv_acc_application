# Local toolchains

This directory is for machine-local cross toolchains and SDK exports.

Expected local layout:

- `vitis-aarch32-gcc/`: copied from `/opt/Xilinx/2025.2/gnu/aarch32/lin/gcc-arm-linux-gnueabi`
- `yocto/`: optional PetaLinux/Yocto SDK export

The actual toolchain directories are intentionally ignored by Git. `../build.sh`
uses `vitis-aarch32-gcc/` first, then falls back to a usable SDK specified by
`SDK_ENV`.
