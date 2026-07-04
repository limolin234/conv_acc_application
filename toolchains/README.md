# Local toolchains

This directory is for machine-local cross toolchains and SDK exports.

Expected local layout:

- `vitis-aarch32-gcc/`: copied from `/opt/Xilinx/2025.2/gnu/aarch32/lin/gcc-arm-linux-gnueabi`
- `yocto/`: optional PetaLinux/Yocto SDK export

The actual toolchain directories are intentionally ignored by Git. `../build.sh`
uses the PetaLinux SDK from `../../external/petalinux/petalinux-sdk/sdk` by
default because it matches the board rootfs. The Vitis GNU toolchain is kept only
as an explicit fallback for non-board smoke builds:

```bash
ALLOW_VITIS_TOOLCHAIN=1 ./build.sh build
```

The copied PetaLinux SDK contains native tools whose ELF interpreter still
points at the original SDK install path. On this machine that compatibility path
is a symlink:

```bash
/home/limolin/petalinux_project/zynq7020_petalinux/images/linux/sdk -> /media/limolin/Disk2/MyProjects/dachuang/hardware/external/petalinux/petalinux-sdk/sdk
```
