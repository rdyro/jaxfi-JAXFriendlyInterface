import sys
from pathlib import Path

import pytest

path = Path(__file__).absolute().parents[1]
if str(path) not in sys.path:
    sys.path.insert(0, str(path))


import jaxfi as jaxm  # noqa: E402


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


@pytest.mark.parametrize("device", ["cpu", "cuda"] if HAS_GPU else ["cpu"])
def test_device_index_notation(device):
    num_of_devices = len(jaxm.jax.devices(device))
    for i in range(num_of_devices):
        device_with_idx = f"{device}:{i}"
        x = jaxm.randn((10, 2), device=device_with_idx)  # by keyword
        # positional device specific in generating functions does not work (nor should it)
        y = jaxm.randn((10, 2), device=device_with_idx)
        z = fn(x, y)
        z_devices = list(z.devices())
        assert len(z_devices) == 1
        assert z_devices[0].device_kind == jaxm.resolve_device(device).device_kind
        assert z_devices[0].id == i

    # check the device past the number of devices, should fail
    i = num_of_devices
    correctly_fails = False
    try:
        jaxm.resolve_device(f"{device}:{i}")
    except: # noqa: E722
        correctly_fails = True
    assert correctly_fails


@pytest.mark.parametrize("device", ["cpu", "cuda"] if HAS_GPU else ["cpu"])
def test_sharding_placement_with_to(device):
    print("device")
    device_list = jaxm.jax.devices(device)
    if len(device_list) <= 1:
        return
    from jax.experimental import mesh_utils
    from jax.sharding import PositionalSharding

    sharding = PositionalSharding(
        mesh_utils.create_device_mesh((len(device_list),), devices=device_list)
    )
    r = jaxm.randn((len(device_list), 17))
    r2 = jaxm.to(r, device=sharding.reshape((-1, 1)))
    r3 = jaxm.to(r, sharding.reshape((-1, 1)))
    print(jaxm.jax.debug.visualize_array_sharding(r))
    print(jaxm.jax.debug.visualize_array_sharding(r2))
    print(jaxm.jax.debug.visualize_array_sharding(r3))
    assert len(r2.devices()) == len(device_list)
    assert len(r3.devices()) == len(device_list)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if HAS_GPU else ["cpu"])
@pytest.mark.parametrize("dtype", [jaxm.float32, jaxm.float64, None])
def test_whether_arrays_are_committed(device, dtype):
    jaxm.set_default_device(device)
    test_commited = lambda x: not hasattr(x, "_committed") or not x._committed # noqa: E731
    assert test_commited(jaxm.randn((10, 2), dtype=dtype))
    assert test_commited(jaxm.rand((10, 2), dtype=dtype))
    assert test_commited(jaxm.randint(0, 10, (10, 2)))
    assert test_commited(jaxm.randperm(10, dtype=dtype))
    assert test_commited(jaxm.ones(10, dtype=dtype))
    assert test_commited(jaxm.zeros(10, dtype=dtype))
    assert test_commited(jaxm.full(10, 1.0, dtype=dtype))
    assert test_commited(jaxm.eye(10, dtype=dtype))
    assert test_commited(jaxm.logspace(0, 10, 20, dtype=dtype))
    assert test_commited(jaxm.arange(10, dtype=dtype))
    assert test_commited(jaxm.linspace(0, 10, 20, dtype=dtype))


####################################################################################################

if __name__ == "__main__":
    for device in ["cpu", "cuda"] if HAS_GPU else ["cpu"]:
        test_device_index_notation(device)
        test_sharding_placement_with_to(device)
        for dtype in [jaxm.float32, jaxm.float64, None]:
            test_whether_arrays_are_committed(device, dtype)
