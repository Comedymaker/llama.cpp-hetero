#!/bin/sh
set -eu
cmake -S /work/backend/htp -B /work/backend/build-dsp -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/work/backend/htp/cmake-toolchain.cmake -DHEXAGON_SDK_ROOT=/opt/hexagon/6.6.0.0 -DHEXAGON_TOOLS_ROOT=/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07 -DDSP_VERSION=v79 -DPREBUILT_LIB_DIR=toolv19_v79
cmake --build /work/backend/build-dsp -j4
