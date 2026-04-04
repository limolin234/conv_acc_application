#!/bin/bash
source /home/limolin/petalinux_project/zynq7020_petalinux/images/linux/sdk/\
environment-setup-cortexa9t2hf-neon-xilinx-linux-gnueabi

exec "$@"
