#!/bin/sh
set -eu
cd /queue/build-android
OBJ=ggml/src/ggml-hexagon/CMakeFiles/ggml-hexagon.dir/ggml-hexagon.cpp.o
COMMAND=$(ninja -t commands "$OBJ" | tail -n1 | sed -e 's|/queue/ggml/src/ggml-hexagon/ggml-hexagon.cpp|/work/backend/ggml-hexagon.cpp|g' -e "s|$OBJ|/work/backend/hexagon.o|g")
sh -c "$COMMAND -MF /work/backend/hexagon.d -I/work/backend -I/queue/ggml/src/ggml-hexagon"
COMMAND=$(ninja -t commands bin/libggml-hexagon.so | tail -n1 | sed -e "s|$OBJ|/work/backend/hexagon.o|g" -e 's|-o bin/libggml-hexagon.so |-o /work/libggml-hexagon.so |g' -e 's|dependency-file=ggml/src/ggml-hexagon/CMakeFiles/ggml-hexagon.dir/link.d|dependency-file=/work/backend/hexagon-link.d|g' -e 's|bin/libggml-base.so|/work/libggml-base.so|g')
sh -c "$COMMAND"
CXX=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++
COMMON="--target=aarch64-none-linux-android34 -std=c++17 -O3 -pthread -I/queue/ggml/include -I/queue/ggml/src -I/queue/ggml/src/ggml-hexagon -I/work -I/work/format -L/work -Wl,-rpath,\$ORIGIN -Wl,--allow-shlib-undefined"
"$CXX" $COMMON /work/convert.cpp -o /work/convert
"$CXX" $COMMON /work/mixprobe.cpp -lggml-hexagon -lggml -lggml-base -lggml-cpu -ldl -o /work/probe
