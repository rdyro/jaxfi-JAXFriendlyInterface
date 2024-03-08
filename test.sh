#!/usr/bin/env bash

env LD_PRELOAD="/usr/local/cuda/lib64/libcudart.so:$LD_PRELOAD" \
  JAXFI_LOAD_SYSTEM_CUDA_LIBS=true \
  pytest tests
