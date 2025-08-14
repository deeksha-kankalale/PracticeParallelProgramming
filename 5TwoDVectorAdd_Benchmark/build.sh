#!/usr/bin/env bash
set -e
nvcc -O3 -arch=sm_87 2DVectorAdd.cu cuda_bench.cu -o matadd_run.out
echo "Built: ./matadd_run"
echo "Example: ./matadd_run"
