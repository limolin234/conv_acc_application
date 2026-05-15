#!/usr/bin/env bash
set -e

source ./env.sh

cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake
cmake --build build -j

scp ./build/DmaApp petalinux@192.168.1.4:/home/petalinux/
ssh petalinux@192.168.1.4 "chmod +x DmaApp"
scp ./build/AccUnitTest petalinux@192.168.1.4:/home/petalinux/
ssh petalinux@192.168.1.4 "chmod +x AccUnitTest"
