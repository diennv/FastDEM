#!/bin/bash
# Build height_mapping benchmarks
#
# Usage:
#   ./build.sh                    # Build all
#   ./build.sh benchmark_height_update   # Build specific
#
# Set BUILD_DIR to override the CMake build directory search, e.g.:
#   BUILD_DIR=/path/to/build ./build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Compiler settings
CXX=${CXX:-g++}
CXXFLAGS="-O2 -std=c++17 -fopenmp"

# Include paths
EIGEN_CFLAGS=$(pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")
INCLUDES=(
    "-I../lib/nanoPCL/include"
    "-I../lib/nanoPCL/thirdparty"
    "-I../lib/grid_map_core/include"
    "-I../include"
    "$EIGEN_CFLAGS"
)

# Locate the CMake build directory.
# Probes: in-source build, colcon workspace (src/FastDEM/fastdem/benchmarks -> build/fastdem)
find_lib() {
    local rel_path=$1
    if [ -n "$BUILD_DIR" ]; then
        echo "$BUILD_DIR/$rel_path"
        return
    fi
    local candidates=(
        "../build/$rel_path"
        "../../build/$rel_path"
        "../../../../build/fastdem/$rel_path"
        "../../../build/$rel_path"
    )
    for candidate in "${candidates[@]}"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return
        fi
    done
}

GRID_MAP_LIB=$(find_lib "lib/grid_map_core/libgrid_map_core_hm.a")
FASTDEM_LIB=$(find_lib "libfastdem.a")

missing=0
if [ -z "$GRID_MAP_LIB" ] || [ ! -f "$GRID_MAP_LIB" ]; then
    echo "ERROR: libgrid_map_core_hm.a not found." >&2
    missing=1
fi
if [ -z "$FASTDEM_LIB" ] || [ ! -f "$FASTDEM_LIB" ]; then
    echo "ERROR: libfastdem.a not found." >&2
    missing=1
fi
if [ "$missing" -eq 1 ]; then
    echo "  Build the project first, then re-run:" >&2
    echo "    cd ../.. && cmake -B build && cmake --build build" >&2
    echo "  Or for a colcon workspace:" >&2
    echo "    colcon build --packages-select fastdem" >&2
    echo "  Or set BUILD_DIR explicitly:" >&2
    echo "    BUILD_DIR=/path/to/build ./build.sh" >&2
    exit 1
fi

YAML_FLAGS=$(pkg-config --cflags --libs yaml-cpp 2>/dev/null || echo "-lyaml-cpp")
SPDLOG_FLAGS=$(pkg-config --cflags --libs spdlog 2>/dev/null || echo "-lspdlog")

LINK_FLAGS="$FASTDEM_LIB $GRID_MAP_LIB $YAML_FLAGS $SPDLOG_FLAGS"

# Build function
build_benchmark() {
    local name=$1
    echo "Building ${name}..."
    $CXX $CXXFLAGS "${INCLUDES[@]}" "${name}.cpp" -o "${name}" $LINK_FLAGS
    echo "Built: ${name}"
}

# Main
if [ $# -eq 0 ]; then
    for cpp in *.cpp; do
        build_benchmark "${cpp%.cpp}"
    done
else
    build_benchmark "$1"
fi

echo ""
echo "Run with: ./benchmark_height_update [path/to/kitti.bin]"
