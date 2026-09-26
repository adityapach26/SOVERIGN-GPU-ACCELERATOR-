#!/usr/bin/env bash
set -e

# Download canonical Netlib MPS instances for Phase 33.1

BENCHMARK_DIR="benchmarks/netlib"
mkdir -p "$BENCHMARK_DIR"

# Download and compile the Netlib emps unpacker
if [ ! -x "$BENCHMARK_DIR/emps" ]; then
    echo "Downloading Netlib emps unpacker..."
    curl -f -sL "https://www.netlib.org/lp/data/emps.c" -o "$BENCHMARK_DIR/emps.c"
    gcc -O2 "$BENCHMARK_DIR/emps.c" -o "$BENCHMARK_DIR/emps"
fi

download_instance() {
    local name=$1
    local url="https://www.netlib.org/lp/data/$name"
    local dest_packed="$BENCHMARK_DIR/${name}_packed"
    local dest_mps="$BENCHMARK_DIR/${name}.mps"

    echo "Downloading $name from $url..."
    if ! curl -f -sL "$url" -o "$dest_packed"; then
        echo "Error: Failed to download $name."
        exit 1
    fi

    echo "Unpacking $name..."
    # The emps utility reads from stdin and writes to stdout
    "$BENCHMARK_DIR/emps" < "$dest_packed" > "$dest_mps"
    
    # Cleanup packed file
    rm -f "$dest_packed"
}

download_instance "afiro"
download_instance "adlittle"

echo "Netlib instances downloaded and unpacked successfully to $BENCHMARK_DIR/"
