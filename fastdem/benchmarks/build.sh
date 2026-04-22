#!/bin/bash
# Build height_mapping benchmarks
#
# Usage:
#   ./build.sh                    # Build all
#   ./build.sh benchmark_height_update   # Build specific

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

# Locate the pre-built grid_map_core library (built via CMake as libgrid_map_core_hm).
# Override by setting BUILD_DIR in the environment.
GRID_MAP_LIB=""
if [ -n "$BUILD_DIR" ]; then
    GRID_MAP_LIB="$BUILD_DIR/lib/grid_map_core/libgrid_map_core_hm.a"
else
    # Probe common layouts: in-source build, parent build, colcon/catkin workspace build
    WS_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"  # ~/stais_ws if src/FastDEM/fastdem/benchmarks
    for candidate in \
        "../build/lib/grid_map_core/libgrid_map_core_hm.a" \
        "../../build/lib/grid_map_core/libgrid_map_core_hm.a" \
        "${WS_ROOT}/build/fastdem/lib/grid_map_core/libgrid_map_core_hm.a" \
        "${WS_ROOT}/build/lib/grid_map_core/libgrid_map_core_hm.a"; do
        if [ -f "$candidate" ]; then
            GRID_MAP_LIB="$candidate"
            break
        fi
    done
fi

if [ -z "$GRID_MAP_LIB" ] || [ ! -f "$GRID_MAP_LIB" ]; then
    echo "ERROR: libgrid_map_core_hm.a not found." >&2
    echo "  Build the project first (cmake --build / colcon build), then re-run." >&2
    echo "  Or set BUILD_DIR to your CMake build directory and re-run:" >&2
    echo "    BUILD_DIR=/path/to/build ./build.sh" >&2
    exit 1
fi

GRID_MAP_FLAGS="$GRID_MAP_LIB"

# Build function
build_benchmark() {
    local name=$1
    echo "Building ${name}..."
    $CXX $CXXFLAGS ${INCLUDES[@]} ${name}.cpp -o ${name} $GRID_MAP_FLAGS
    echo "Built: ${name}"
}

# Main
if [ $# -eq 0 ]; then
    # Build all benchmarks
    for cpp in *.cpp; do
        name="${cpp%.cpp}"
        build_benchmark "$name"
    done
else
    # Build specific benchmark
    build_benchmark "$1"
fi

echo ""
echo "Run with: ./benchmark_height_update [path/to/kitti.bin]"
