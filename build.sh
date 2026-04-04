#!/usr/bin/env bash
set -e

source ./env.sh

cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake
cmake --build build -j
