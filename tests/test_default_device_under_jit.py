import sys
from pathlib import Path

import pytest

path = Path(__file__).absolute().parents[1]
if str(path) not in sys.path:
    sys.path.insert(0, str(path))


from jaxfi import jaxm # noqa: E402


def fn(x, y):
    z = x * jaxm.arange(x.size, dtype=x.dtype).reshape(x.shape) - y
    z = z * jaxm.ones(z.shape, dtype=z.dtype)
    z = z + jaxm.zeros(z.shape, dtype=z.dtype)
    z = z * jaxm.linspace(0, 10, z.size, dtype=z.dtype).reshape(z.shape)
    z = z * jaxm.logspace(0, 10, z.size, dtype=z.dtype).reshape(z.shape)
    z = z + jaxm.array(x[0], dtype=x.dtype)
    return z

try:
    jaxm.resolve_device("cuda")
    HAS_GPU = True
except RuntimeError:
    HAS_GPU = False

####################################################################################################

@pytest.mark.parametrize("device", ["cpu", "cuda"] if HAS_GPU else ["cpu"])
@pytest.mark.parametrize("dtype", [jaxm.float32, jaxm.float64])
def test_non_jit_behavior(device, dtype):
    shape = (10, 2)

    x = jaxm.randn(shape, dtype=dtype, device=device)
    y = jaxm.randn(shape, dtype=dtype, device=device)
    z =  fn(x, y)
    z_devices = list(z.devices())
    assert len(z_devices) == 1
    assert z_devices[0].device_kind == jaxm.resolve_device(device).device_kind
    assert z.dtype == dtype

@pytest.mark.parametrize("device", ["cpu", "cuda"] if HAS_GPU else ["cpu"])
@pytest.mark.parametrize("dtype", [jaxm.float32, jaxm.float64])
def test_jit_behavior(device, dtype):
    shape = (10, 2)

    fn_jit = jaxm.jit(fn)

    x = jaxm.randn(shape, dtype=dtype, device=device)
    y = jaxm.randn(shape, dtype=dtype, device=device)
    z =  fn_jit(x, y)
    z_devices = list(z.devices())
    assert z_devices[0].device_kind == jaxm.resolve_device(device).device_kind
    assert z.dtype == dtype


if __name__ == "__main__":
    test_non_jit_behavior()
    test_jit_behavior()
