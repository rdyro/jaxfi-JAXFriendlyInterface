import sys
from pathlib import Path

path = Path(__file__).absolute().parents[1]
if str(path) not in sys.path:
    sys.path.insert(path, 0)


from jfi import jaxm


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

def test_non_jit_behavior():
    devices = [jaxm.resolve_device(x) for x in (["cpu", "cuda"] if HAS_GPU else ["cpu"])]
    shape = (10, 2)

    for dtype in [jaxm.float32, jaxm.float64]:
        for device in devices:
            x = jaxm.randn(shape, dtype=dtype, device=device)
            y = jaxm.randn(shape, dtype=dtype, device=device)
            z =  fn(x, y)
            assert z.device().device_kind == device.device_kind
            assert z.dtype == dtype

def test_jit_behavior():
    devices = [jaxm.resolve_device(x) for x in (["cpu", "cuda"] if HAS_GPU else ["cpu"])]
    shape = (10, 2)

    fn_jit = jaxm.jit(fn)

    for dtype in [jaxm.float32, jaxm.float64]:
        for device in devices:
            x = jaxm.randn(shape, dtype=dtype, device=device)
            y = jaxm.randn(shape, dtype=dtype, device=device)
            z =  fn_jit(x, y)
            assert z.device().device_kind == device.device_kind
            assert z.dtype == dtype


if __name__ == "__main__":
    test_non_jit_behavior()
    test_jit_behavior()