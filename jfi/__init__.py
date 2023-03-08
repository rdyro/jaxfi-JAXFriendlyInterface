import os
import re
import time
import hashlib
import pickle
from typing import Tuple, Any, Optional
from subprocess import check_output

####################################################################################################

jaxm = None
jax, jnp, jsp, jrandom = None, None, None, None
DEFAULT_DEVICE, DEFAULT_DTYPE = None, None

####################################################################################################


def _resolve_device(device: Tuple[str, Any, None], idx: int = 0):
    """Convert device name to the device handle."""
    global jax
    if device is None:
        return DEFAULT_DEVICE
    return jax.devices(device)[idx] if isinstance(device, str) else device


def _resolve_dtype(dtype):
    """Sanitize and check (via error raised) if argument is a dtype specification."""
    global jnp
    return jnp.dtype(dtype)


def _default_dtype_for_device(device):
    """Convert a device to its default dtype (global dtype for CPU, float32 for GPU)."""
    global jnp
    device = _resolve_device(device)
    if re.search("cpu", device.device_kind) is not None:
        return DEFAULT_DTYPE
    else:
        return jnp.float32


def _jaxm_to(
    x: "Array", # noqa: F821
    device_or_dtype: Optional[Tuple[str, Any]] = None,
    device: Optional[Tuple[str, Any]] = None,
    dtype: Optional[Tuple[str, Any]] = None,
):
    """Move an array to the specified device and/or dtype."""
    global jnp, jax
    if device_or_dtype is not None:
        # user specifies only one argument - either a device or dtype
        try:
            dtype = _resolve_dtype(device_or_dtype)
            return x.astype(dtype)
        except TypeError:
            device = _resolve_device(device_or_dtype)
            return jax.device_put(x, device)
    else:
        # user specifies keyword arguments for device and dtype explicitly (via kwargs)
        if device is not None and dtype is not None:
            return jax.device_put(x, _resolve_device(device)).astype(_resolve_dtype(dtype))
        if device is not None and dtype is None:
            return jax.device_put(x, _resolve_device(device)).astype(
                _default_dtype_for_device(device)
            )
        elif device is None and dtype is not None:
            return x.astype(_resolve_dtype(dtype))
        else:
            return x


####################################################################################################


def get_default_dtype():
    """Returns the default dtype for CPU only, default dtype for GPU cannot be changed (float32)."""
    global DEFAULT_DTYPE
    return DEFAULT_DTYPE


def set_default_dtype(dtype):
    """Sets the default dtype for CPU only, default dtype for GPU cannot be changed (float32)."""
    global jnp, DEFAULT_DTYPE
    DEFAULT_DTYPE = _resolve_dtype(dtype)


def get_default_device():
    """Returns the default device for which to create arrays on."""
    global DEFAULT_DEVICE
    return DEFAULT_DEVICE


def set_default_device(device):
    """Sets the default device for which to create arrays on."""
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = device


def manual_seed(val):
    """Sets the random seed for the random number generator."""
    assert jrandom is not None
    global key
    key = jrandom.PRNGKey(val)


####################################################################################################
def init(seed=None):
    """Initializes the wrapped jax backend setting the platform (e.g., GPU) and random seed."""

    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(False)
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    try:
        check_output("nvidia-smi")
        gpu_available = True
    except FileNotFoundError:
        gpu_available = False
    os.environ["JAX_PLATFORM_NAME"] = "GPU" if gpu_available else "CPU"

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
    seed = (
        seed
        if seed is not None
        else hash(
            hashlib.sha256(pickle.dumps((time.time(), os.getpid(), os.urandom(100)))).hexdigest()
        )
    )
    key = jrandom.PRNGKey(seed)

    def device_dtype_fn(fn):
        def fn_(*args, **kw):
            device = _resolve_device(kw.get("device", DEFAULT_DEVICE))
            if "device" in kw:
                del kw["device"]
            kw["dtype"] = kw.get("dtype", _default_dtype_for_device(device))
            return jax.device_put(fn(*args, **kw), device)

        return fn_

    jaxm.ones = device_dtype_fn(jnp.ones)
    jaxm.zeros = device_dtype_fn(jnp.zeros)
    jaxm.full = device_dtype_fn(jnp.full)
    jaxm.eye = device_dtype_fn(jnp.eye)

    def random_fn(fn, first_arg_to_tuple=False):
        def jaxm_fn(*args, **kw):
            global key
            key1, key2 = jrandom.split(key)
            # set correct device and dtype
            device = _resolve_device(kw.get("device", DEFAULT_DEVICE))
            if "device" in kw:
                del kw["device"]
            kw["dtype"] = kw.get("dtype", _default_dtype_for_device(device))
            # make the size a tuple if this is the only positional argument
            if first_arg_to_tuple and len(args) == 1 and not hasattr(args[0], "__iter__"):
                args = [(args[0],)] + list(args)[1:]
            ret = jax.device_put(fn(key1, *args, **kw), device)
            key = key2
            return ret

        return jaxm_fn

    def make_random_keys(n):
        global key
        keys = jrandom.split(key, n + 1)
        key = keys[-1]
        return keys[:-1]

    jaxm.randn = random_fn(jrandom.normal, first_arg_to_tuple=True)
    jaxm.rand = random_fn(jrandom.uniform, first_arg_to_tuple=True)
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
    jaxm.get_default_dtype = get_default_dtype
    jaxm.set_default_dtype = set_default_dtype
    jaxm.get_default_device = get_default_device
    jaxm.set_default_device = set_default_device
    jaxm.make_random_keys = make_random_keys
    jaxm.to = _jaxm_to

    # module bindings
    jaxm.jax = jax
    jaxm.numpy = jnp
    jaxm.lax = jax.lax
    jaxm.xla = jax.xla
    jaxm.scipy = jsp
    jaxm.random = jrandom

    return jaxm


jaxm = init()
