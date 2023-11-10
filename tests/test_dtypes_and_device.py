import re

import sys
from pathlib import Path
from warnings import warn
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

path = Path(__file__).absolute().parents[1]
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from jaxfi import jaxm
import jaxfi

def _allocate(opts=None):
    opts = dict() if opts is None else opts
    return {
        "ones": jaxm.ones(2, **opts),
        "zeros": jaxm.zeros(2, **opts),
        "full": jaxm.full(2, 2.0, **opts),
        "eye": jaxm.eye(2, **opts),
        "randn": jaxm.randn(2, **opts),
        "rand": jaxm.rand(2, **opts),
    }

def _check_correct_placement(zs, dtype, device):
    for k, z in zs.items():
        msg = f"failed for {k}"
        assert z.dtype == dtype, msg
        if device.lower() == "cpu":
            cond = re.search("cpu", z.device().device_kind, flags=re.IGNORECASE) is not None
        else:
            cond = re.search("cpu", z.device().device_kind, flags=re.IGNORECASE) is None
        if int(getattr(jaxm.jax, "__version__", "0.0.0").split(".")[1]) < 4:
            if not cond:
                msg = (
                    "default device does not appear to be working, "
                    + "this is expected on jax.__version__ < 0.4.0"
                )
                logger.warning(msg)
        else:
            assert cond, msg


def test_construction():
    jaxm.set_default_device("cpu")
    jaxm.set_default_dtype(jaxm.float64)
    # cpu first
    _check_correct_placement(_allocate(), jaxm.float64, "cpu")
    # float32 on cpu next
    jaxm.set_default_dtype(jaxm.float32)
    _check_correct_placement(_allocate(), jaxm.float32, "cpu")
    # gpu, if available
    try:
        jaxm.jax.devices("gpu")
        has_gpu = True
    except RuntimeError:
        has_gpu = False
    if has_gpu:
        jaxm.set_default_device("gpu")
        _check_correct_placement(_allocate(), jaxm.float32, "gpu")

    # test if CPU allocation still works
    topts = dict(device="cpu")
    _check_correct_placement(_allocate(topts), jaxm.float32, "cpu")
    jaxm.set_default_dtype(jaxm.float64)
    _check_correct_placement(_allocate(topts), jaxm.float64, "cpu")
    jaxm.set_default_device("cpu")
    jaxm.set_default_dtype(jaxm.float64)


def test_moving():
    try:
        jaxm.jax.devices("gpu")
        dtypes, devices = [jaxm.float32, jaxm.float64], ["cpu", "gpu"]
    except RuntimeError:
        dtypes, devices = [jaxm.float32, jaxm.float64], ["cpu"]

    def check_dtype_device(x, dtype=None, device=None):
        if dtype is not None:
            assert x.dtype == dtype
        if device is not None:
            assert x.device() == device

    for dtype in dtypes:
        for device in devices:
            dtype, device = jaxfi.resolve_dtype(dtype), jaxfi.resolve_device(device)
            r = jaxm.randn(2, dtype=dtype, device=device)
            check_dtype_device(r, dtype=dtype, device=device)

            r = jaxm.randn(2, dtype=dtype)
            check_dtype_device(r, dtype=dtype)
            r = jaxm.randn(2, device=device)
            check_dtype_device(r, device=device)

            r = jaxm.to(jaxm.randn(2), dtype)
            check_dtype_device(r, dtype=dtype)
            r = jaxm.to(jaxm.randn(2), device)
            check_dtype_device(r, device=device)

            r = jaxm.to(jaxm.randn(2), dtype=dtype)
            check_dtype_device(r, dtype=dtype)
            r = jaxm.to(jaxm.randn(2), device=device)
            check_dtype_device(r, device=device)

            r = jaxm.to(jaxm.randn(2), dtype=dtype, device=device)
            check_dtype_device(r, dtype=dtype, device=device)


if __name__ == "__main__":
    test_construction()
    test_moving()
