# 2D Matrix Add (CUDA) + Micro-Benchmark

**What it is:** A tiny, reusable CUDA benchmarking harness that times a 2D matrix add kernel and reports **kernel-only** performance (avg ms, GB/s, GFLOP/s).

## Quick Start
```bash
./build.sh
./matadd_run        # defaults 100x100
./matadd_run 4096 4096

