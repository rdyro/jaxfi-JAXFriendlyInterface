import os
import re
import time
from typing import Tuple, Any, Optional

import numpy as np

jaxm = None
jax, jnp, jsp, jrandom = None, None, None, None
DEFAULT_DEVICE, DEFAULT_DTYPE = None, None


def manual_seed(val):
    assert jrandom is not None
    global key
    key = jrandom.PRNGKey(val)


def resolve_device(device: Tuple[str, Any, None], idx: int = 0):
    global jax
    if device is None:
        return DEFAULT_DEVICE
    return jax.devices(device)[idx] if isinstance(device, str) else device


def get_default_dtype(device):
    global jnp
    device = resolve_device(device)
    if re.search("cpu", device.device_kind) is not None:
        return jnp.float64
    else:
        return jnp.float32


def jaxm_to(
    x: "Array",
    device_or_dtype: Optional[Tuple[str, Any]] = None,
    device: Optional[Tuple[str, Any]] = None,
    dtype: Optional[Tuple[str, Any]] = None,
):
    global jnp, jax
    if device_or_dtype is not None:
        try:
            dtype = jnp.dtype(device_or_dtype)
            return x.astype(dtype)
        except TypeError:
            device = resolve_device(device_or_dtype)
            return jax.device_put(x, device)
    else:
        if device is not None and dtype is not None:
            return jax.device_put(x, resolve_device(device)).astype(jnp.dtype(dtype))
        if device is not None and dtype is None:
            return jax.device_put(x, resolve_device(device)).astype(get_default_dtype(device))
        elif device is None and dtype is not None:
            return x.astype(jnp.dtype(dtype))
        else:
            return x
        


def set_default_dtype(dtype):
    global jnp, DEFAULT_DTYPE
    DEFAULT_DTYPE = jnp.dtype(dtype)


def init(seed=None):
    os.environ["JAX_PLATFORM_NAME"] = "GPU"
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(False)
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    global jax, jnp, jsp, jrandom, DEFAULT_DEVICE, DEFAULT_DTYPE
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import jax.scipy as jsp

    # binding main derivatives and jit
    jaxm = jnp
    jaxm.grad, jaxm.jacobian = jax.grad, jax.jacobian
    jaxm.jvp, jaxm.vjp = jax.jvp, jax.vjp
    jaxm.hessian = jax.hessian
    jaxm.jit = jax.jit
    jaxm.vmap = jax.vmap

    DEFAULT_DEVICE, DEFAULT_DTYPE = jax.devices("cpu")[0], jnp.float64

    # binding random numbers
    global key
    key = jrandom.PRNGKey(int(time.time()) if seed is None else seed)

    def device_dtype_fn(fn):
        def fn_(*args, **kw):
            device = resolve_device(kw.get("device", DEFAULT_DEVICE))
            if "device" in kw:
                del kw["device"]
            kw["dtype"] = kw.get("dtype", get_default_dtype(device))
            return jax.device_put(fn(*args, **kw), device)

        return fn_

    jaxm.ones = device_dtype_fn(jnp.ones)
    jaxm.zeros = device_dtype_fn(jnp.zeros)
    jaxm.eye = device_dtype_fn(jnp.eye)
    jaxm.eye = device_dtype_fn(jnp.eye)

    def random_fn(fn):
        def jaxm_fn(*args, **kw):
            global key
            key1, key2 = jrandom.split(key)
            device = resolve_device(kw.get("device", DEFAULT_DEVICE))
            if "device" in kw:
                del kw["device"]
            kw["dtype"] = kw.get("dtype", get_default_dtype(device))
            ret = jax.device_put(fn(key1, *args, **kw), device)
            key = key2
            return ret

        return jaxm_fn

    def make_random_keys(n):
        global key
        keys = jrandom.split(key, n + 1)
        key = keys[-1]
        return keys[:-1]

    jaxm.randn = random_fn(jrandom.normal)
    jaxm.rand = random_fn(jrandom.uniform)
    jaxm.randint = random_fn(lambda key, low, high, size: jrandom.randint(key, size, low, high))

    # LA factorizations and solves
    jaxm.linalg.cholesky = jsp.linalg.cho_factor
    jaxm.linalg.cholesky_solve = jsp.linalg.cho_solve
    jaxm.linalg.lu_factor = jsp.linalg.lu_factor
    jaxm.linalg.lu_solve = jsp.linalg.lu_solve

    # some utility bindings
    jaxm.norm = jnp.linalg.norm
    jaxm.softmax = jax.nn.softmax
    jaxm.cat = jnp.concatenate
    jaxm.t = lambda x: jaxm.swapaxes(x, -1, -2)
    jaxm.nn = jax.nn
    jaxm.manual_seed = manual_seed
    jaxm.make_random_keys = make_random_keys
    jaxm.to = jaxm_to

    # module bindings
    jaxm.jax = jax
    jaxm.numpy = jnp
    jaxm.lax = jax.lax
    jaxm.xla = jax.xla
    jaxm.scipy = jsp
    jaxm.random = jrandom

    return jaxm


jaxm = init()
