#!/bin/bash

set -e

COMPILE=$1  

if [ $COMPILE -gt 0 ] ; then
    echo "Compiling iLQR_ADMM.cu"
    nvcc -std=c++11 -o iLQR.exe iLQR_ADMM.cu utils/cudaUtils.cu utils/threadUtils.cpp -gencode arch=compute_61,code=sm_61 -rdc=true -O3
fi

echo "Running iLQR.exe"
./iLQR.exe

echo "Running pendulum_follow.py"
python3 pendulum_sim/pendulum_follow.py