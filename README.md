# GPU_accelerated_ADMM

# Compiling and Running Code
Compile using nvcc : nvcc -std=c++11 -o iLQR.exe WAFR_iLQR_examples.cu utils/cudaUtils.cu utils/threadUtils.cpp -gencode arch=compute_61,code=sm_61 -rdc=true -O3

run: ./iLQR.exe G

source drake-setup/env/bin/activate

python3 sim/play_pydrake.py

#install drake using these instructions first in drake-setup folder or wherever
https://drake.mit.edu/pip.html#stable-releases

python3 -m venv env
env/bin/pip install --upgrade pip
env/bin/pip install drake
source env/bin/activate