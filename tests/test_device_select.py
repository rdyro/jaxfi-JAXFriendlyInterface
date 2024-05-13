import re

import sys
from pathlib import Path
import logging
import pytest

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

path = Path(__file__).absolute().parents[1]
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from jaxfi import jaxm  # noqa: E402
import jaxfi  # noqa: E402
from jax import Array
from jaxfi.experimental.device_cond import device_select

try:
    jaxm.jax.devices("gpu")
    HAS_GPU = True
except RuntimeError:
    HAS_GPU = False


def test_device_select():
    if not HAS_GPU:
        return

    def cpu_fn(x: Array) -> Array:
        print("Hello from CPU")
        return dict(a=x)

    def gpu_fn(x: Array) -> Array:
        y = x * 2 
        s = jaxm.exp(y)
        s = jaxm.sum(s, axis=0)
        print("Hello from GPU")
        return dict(a=2 * x - s)  # (2 * x,)

    @jaxm.jit
    def test_fn(x):
        return device_select(cpu_fn, gpu_fn)(x)

    x = jaxm.randn((2, 3), device="cpu")
    y = jaxm.to(x, device="gpu")

    print(test_fn(x))
    print(test_fn(y))

    print(test_fn.lower(x).as_text())
    print("#" * 80)
    print(test_fn.lower(y).as_text())

if __name__ == "__main__":
    test_device_select()