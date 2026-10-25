#!/bin/sh
set -eu
CXX=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++
"$CXX" --target=aarch64-none-linux-android34 -std=c++17 -O3 -DNDEBUG -fPIC -shared -pthread /work/staged-preload.cpp -ldl -o /work/libstaged-q23.so
