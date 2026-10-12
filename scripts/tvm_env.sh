#!/usr/bin/env bash
# Source this file from /home/huriyeh.babak/s4convd-paper before TVM runs.

TVM_ENV="${TVM_ENV:-$HOME/tvm-build-env-v025}"
TVM_SRC="${TVM_SRC:-$HOME/third_party/apache-tvm-v0.25.0}"
TVM_BUILD="${TVM_BUILD:-$TVM_SRC/build-cuda-release}"
CUDA_ROOT="${CUDA_ROOT:-$HOME/micromamba/envs/s4convd}"

export TVM_ENV TVM_SRC TVM_BUILD CUDA_ROOT
export PYTHONPATH="$TVM_SRC/python${PYTHONPATH:+:$PYTHONPATH}"
export TVM_LIBRARY_PATH="$TVM_BUILD/lib"
export LD_LIBRARY_PATH="$TVM_BUILD/lib:$TVM_ENV/lib:$CUDA_ROOT/lib:$CUDA_ROOT/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$TVM_ENV/bin:$CUDA_ROOT/bin:$PATH"

if [[ ! -x "$TVM_ENV/bin/python" ]]; then
  echo "Missing TVM Python: $TVM_ENV/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -f "$TVM_BUILD/lib/libtvm_compiler.so" ]]; then
  echo "Missing TVM compiler library: $TVM_BUILD/lib/libtvm_compiler.so" >&2
  return 1 2>/dev/null || exit 1
fi
