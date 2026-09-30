# LAKSHYA

**Indigenous GPU-Accelerated Optimization Solver**

LAKSHYA is a high-performance, GPU-native mathematical optimization engine. Designed for research and industrial-scale problems, it leverages massively parallel architecture to solve complex Linear Programming (LP) and related optimization models with speed and precision.

## Key Features

* **Massively Parallel GPU Execution**: Native CUDA and cuSPARSE integration for ultra-fast matrix operations.
* **Interior Point Method (IPM)**: Implements the robust Mehrotra Predictor-Corrector algorithm optimized for GPU execution.
* **Advanced KKT Solver**: Employs iterative refinement, Markowitz-threshold pivoting Sparse LU, and GPU-accelerated Cholesky factorization.
* **Bulletproof Robustness**: Seamless CPU Phase-1 Simplex fallback recovery mechanism guarantees an optimal solution even in cases of severe numerical instability or ill-conditioned matrices (e.g., standard Netlib benchmarks like `adlittle`).
* **LAKSHYA Engineering Console**: A polished, interactive, dark-themed terminal UI providing real-time telemetry, live solver iterations, and certificate verification without sacrificing engine performance.

## System Architecture

* `src/cuda/` - GPU-accelerated KKT solvers, memory management (VRAM arena), and custom CUDA kernels.
* `src/ipm/` - Core Interior Point Method logic (Mehrotra device solver).
* `src/numerics/` - Robust numerical routines including perturbation and Sparse LU factorization.
* `src/simplex/` - CPU-based Primal/Dual Simplex algorithms used for Phase-1 basis recovery.
* `src/core/` - Sparse matrix representations (CSR/CSC) and core data types.
* `app.py` - Flask-based backend for the LAKSHYA Engineering Console.

## Requirements

* **Compiler**: C++17 compatible compiler (e.g., GCC 7+, Clang 5+, MSVC 2017+)
* **Build System**: CMake 3.18 or higher
* **GPU Computing**: NVIDIA CUDA Toolkit (includes `nvcc`, `cuSPARSE`, and `Thrust`)
* **Python Environment** (for the UI): Python 3.8+ with `Flask` installed (`pip install Flask`)

## Building the Solver

The project uses CMake for its build system. It requires a compatible C++17 compiler and the NVIDIA CUDA Toolkit.

```bash
# Clone the repository
git clone https://github.com/adityapach26/SOVERIGN-GPU-ACCELERATOR-.git
cd SOVERIGN-GPU-ACCELERATOR-

# Generate build files
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build build -j$(nproc)
```

## Running the Benchmark CLI

You can run standard `.mps` benchmark files directly through the command-line interface:

```bash
./build/tests/sankhya_benchmark_cli benchmarks/netlib/afiro.mps
```

The CLI outputs standard telemetry in JSON format, capturing solver status, primal residuals, iterations, and execution path (GPU-only vs. GPU+CPU Recovery).

## LAKSHYA Interactive UI

To launch the LAKSHYA Engineering Console:

1. Ensure the Python environment is set up (Flask required).
2. Start the server:
   ```bash
   python app.py
   ```
3. Open a web browser and navigate to the provided local URL (typically `http://localhost:5000`).
4. Upload an MPS file (e.g., `adlittle.mps`) to watch the real-time GPU optimization process.

## License

Proprietary / All Rights Reserved.
