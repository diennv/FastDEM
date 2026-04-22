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

# Locate the pre-built grid_map_core library (built via CMake as libgrid_map_core_hm)
GRID_MAP_LIB=""
for candidate in \
    "../build/lib/grid_map_core/libgrid_map_core_hm.a" \
    "../../build/lib/grid_map_core/libgrid_map_core_hm.a"; do
    if [ -f "$candidate" ]; then
        GRID_MAP_LIB="$candidate"
        break
    fi
done

if [ -z "$GRID_MAP_LIB" ]; then
    echo "ERROR: libgrid_map_core_hm.a not found. Run cmake --build from the project root first." >&2
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
