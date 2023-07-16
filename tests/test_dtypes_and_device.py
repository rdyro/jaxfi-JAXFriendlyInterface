import re

import sys
from pathlib import Path

path = Path(__file__).absolute().parents[1]
if str(path) not in sys.path:
    sys.path.insert(path, 0)

from jfi import jaxm
import jfi


def test_construction():
    jaxm.set_default_device("cpu")
    jaxm.set_default_dtype(jaxm.float64)
    # cpu first
    zs = [
        jaxm.ones(2),
        jaxm.zeros(2),
        jaxm.full(2, 2.0),
        jaxm.eye(2),
        jaxm.randn(2),
        jaxm.rand(2),
    ]
    for z in zs:
        assert z.dtype == jaxm.float64
        assert re.search("cpu", z.device().device_kind, flags=re.IGNORECASE) is not None
    # float32 on cpu next
    jaxm.set_default_dtype(jaxm.float32)
    zs = {
        "ones": jaxm.ones(2),
        "zeros": jaxm.zeros(2),
        "full": jaxm.full(2, 2.0),
        "eye": jaxm.eye(2),
        "randn": jaxm.randn(2),
        "rand": jaxm.rand(2),
    }
    for k, z in zs.items():
        msg = f"failed for {k}"
        assert z.dtype == jaxm.float32, msg
        assert re.search("cpu", z.device().device_kind, flags=re.IGNORECASE) is not None, msg
    # gpu, if available
    try:
        jaxm.jax.devices("gpu")
        has_gpu = True
    except RuntimeError:
        has_gpu = False
    if has_gpu:
        jaxm.set_default_device("gpu")
        zs = [
            jaxm.ones(2),
            jaxm.zeros(2),
            jaxm.full(2, 2.0),
            jaxm.eye(2),
            jaxm.randn(2),
            jaxm.rand(2),
        ]
        for z in zs:
            assert z.dtype == jaxm.float32
            assert re.search("cpu", z.device().device_kind, flags=re.IGNORECASE) is None
    # test if CPU allocation still works
    topts = dict(device="cpu")
    zs = [
        jaxm.ones(2, **topts),
        jaxm.zeros(2, **topts),
        jaxm.full(2, 2.0, **topts),
        jaxm.eye(2, **topts),
        jaxm.randn(2, **topts),
        jaxm.rand(2, **topts),
    ]
    for z in zs:
        assert z.dtype == jaxm.float32
        assert re.search("cpu", z.device().device_kind, flags=re.IGNORECASE) is not None
    jaxm.set_default_dtype(jaxm.float64)
    zs = [
        jaxm.ones(2, **topts),
        jaxm.zeros(2, **topts),
        jaxm.full(2, 2.0, **topts),
        jaxm.eye(2, **topts),
        jaxm.randn(2, **topts),
        jaxm.rand(2, **topts),
    ]
    for z in zs:
        assert z.dtype == jaxm.float64
        assert re.search("cpu", z.device().device_kind, flags=re.IGNORECASE) is not None
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
            dtype, device = jfi.resolve_dtype(dtype), jfi.resolve_device(device)
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
