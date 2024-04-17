import sys
import os
from pathlib import Path
import ctypes
from ctypes.util import find_library

import sysconfig


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CUDA_LIB_ALLOW_LIST = [
    "cudart",
    "cublas",
    "cublasLt",
    "cufft",
    "cufftw",
    "cufile",
    "cufile_rdma",
    "cuinj64",
    "curand",
    "cusolver",
    "cusolverMg",
    "cusparse",
    "cudnn",
    "cudnn_ops_train",
    "cudnn_ops_infer",
    "cudnn_adv_train",
    "cudnn_adv_infer",
    "cudnn_cnn_train",
    "cudnn_cnn_infer",
]

CUDA_LIBS_TO_FIND = [
    "cupti",
]

ARE_CUDA_LIBRARIES_LOADED = False

####################################################################################################

def _get_shsuffix():
    suffix = sysconfig.get_config_var('SHLIB_SUFFIX')
    if suffix is None:  # Some platforms may not have this set
        if sys.platform == 'win32':
            suffix = '.dll'
        elif sys.platform.startswith('linux'):
            suffix = '.so'
        elif sys.platform == 'darwin':
            suffix = '.dylib'
    return suffix

def _find_custom_cuda_lib(libname: str):
    candidates = []
    for root, dir, files in os.walk("/usr/local/cuda"):
        matching_files = [file for file in files if file.startswith(libname)]
        if len(matching_files) == 0:
            continue
        candidates.append(Path(root) / sorted(matching_files, key=lambda x: len(x))[0])
    assert len(candidates) > 0, f"Could not find CUDA library {libname}"
    return sorted(candidates, key=lambda x: len(str(x)))[0]


def load_cuda_libs():
    """This function uses ctypes to load OS preferred CUDA libraries.
    This should allow JAX to use system CUDA, rather than the one bundled with
    another python package, e.g. PyTorch."""

    assert sys.platform.startswith("linux"), "This function is only supported on Linux"

    global ARE_CUDA_LIBRARIES_LOADED
    if ARE_CUDA_LIBRARIES_LOADED:
        return
    logger.info("Loading CUDA libraries...")
    for lib in CUDA_LIB_ALLOW_LIST:
        lib_name = find_library(lib)
        if lib_name is not None:
            logger.info(f"Dynamically loading CUDA library: {lib_name}")
            try:
                ctypes.CDLL(lib_name)
            except: # noqa E722
                logger.info(f"Failed to load {lib_name}")
    for lib in CUDA_LIBS_TO_FIND:
        lib_name = _find_custom_cuda_lib(f"lib{lib}" + _get_shsuffix())
        logger.info(f"Dynamically loading CUDA library: {lib_name}")
        try:
            ctypes.CDLL(lib_name)
        except: # noqa E722
            logger.info(f"Failed to load {lib_name}")
    ARE_CUDA_LIBRARIES_LOADED = True
