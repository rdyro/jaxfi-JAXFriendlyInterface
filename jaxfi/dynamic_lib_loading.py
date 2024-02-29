import ctypes
from ctypes.util import find_library

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

ARE_CUDA_LIBRARIES_LOADED = False


def load_cuda_libs():
    """This function uses ctypes to load OS preferred CUDA libraries.
    This should allow JAX to use system CUDA, rather than the one bundled with
    another python package, e.g. PyTorch."""

    global ARE_CUDA_LIBRARIES_LOADED
    if ARE_CUDA_LIBRARIES_LOADED:
        return
    logger.info("Loading CUDA libraries...")
    for lib in CUDA_LIB_ALLOW_LIST:
        lib_name = find_library(lib)
        if lib_name is not None:
            logger.info(f"Dynamically loading CUDA library: {lib_name}")
            ctypes.CDLL(lib_name)
    ARE_CUDA_LIBRARIES_LOADED = True
