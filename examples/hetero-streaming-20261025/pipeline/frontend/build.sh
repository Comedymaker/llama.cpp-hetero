#!/bin/sh
set -eu
CXX=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++
"$CXX" --target=aarch64-none-linux-android34 -std=c++17 -O3 -DNDEBUG -pthread -I/queue/include -I/queue/src -I/queue/ggml/include -I/work -I/queue/examples/layerskip /work/frontend.cpp -L/work '-Wl,-rpath,$ORIGIN' -Wl,--export-dynamic -Wl,--allow-shlib-undefined -lllama -lggml -lggml-base -lggml-cpu -ldl -lm -o /work/llama-batch-gain-plannedinitial
