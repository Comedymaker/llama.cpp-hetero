#!/bin/sh
set -eu
cd /queue/build-android
OBJ=src/CMakeFiles/llama.dir/llama-context.cpp.o
COMMAND=$(ninja -t commands "$OBJ" | tail -n1 | sed -e 's|/queue/src/llama-context.cpp|/work/host/llama-context.cpp|g' -e "s|$OBJ|/work/host/context.o|g")
sh -c "$COMMAND -MF /work/host/context.d -I/queue/src -I/queue/ggml/src -I/work/host -DLLAMA_TREE_OLD_MODEL_OBJECTS"
COMMAND=$(ninja -t commands bin/libllama.so | tail -n1 | sed -e "s|$OBJ|/work/host/context.o|g" -e 's|src/CMakeFiles/llama.dir/models/llama.cpp.o|/work/host/model-llama.o|g' -e 's|src/CMakeFiles/llama.dir/llama-model.cpp.o|src/CMakeFiles/llama.dir/llama-model-quant-direct.o|g' -e 's|src/CMakeFiles/llama.dir/llama-model-loader.cpp.o|src/CMakeFiles/llama.dir/llama-model-loader-quant-direct.o|g' -e 's|src/CMakeFiles/llama.dir/llama-mmap.cpp.o|src/CMakeFiles/llama.dir/llama-mmap-quant-direct.o|g' -e 's|-o bin/libllama.so |-o /work/host/libllama.so |g' -e 's|dependency-file=src/CMakeFiles/llama.dir/link.d|dependency-file=/work/host/link.d|g')
sh -c "$COMMAND"
