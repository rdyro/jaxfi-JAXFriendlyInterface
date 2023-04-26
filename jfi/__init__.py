import os
import re
import copyreg
import time
import hashlib
import pickle
from typing import Tuple, Any, Optional
from subprocess import check_output
import numpy as np

ModuleType = type(os)  # a Python version agnostic to the get the module type

####################################################################################################

jaxm = None
jax, jnp, jsp, jrandom = None, None, None, None
DEFAULT_DEVICE, DEFAULT_DTYPE = None, None

####################################################################################################


def make_random_keys(n):
    global key, jrandom
    keys = jrandom.split(key, n + 1)
    key = keys[-1]
    return keys[:-1]


def make_random_key():
    return make_random_keys(1)[0]


def copy_module(mod):
    new_mod = ModuleType(mod.__name__ + "_jfi")
    for attr in dir(mod):
        if not attr.startswith("_"):
            if isinstance(getattr(mod, attr), ModuleType):
                setattr(new_mod, attr, copy_module(getattr(mod, attr)))
            else:
                setattr(new_mod, attr, getattr(mod, attr))
    return new_mod


####################################################################################################


def _is_dtype(x):
    global jnp
    try:
        jnp.dtype(x)
        return True
    except TypeError:
        return False


def resolve_device(device: Tuple[str, Any, None], idx: int = 0):
    """Convert device name to the device handle."""
    global jax
    if device is None:
        return DEFAULT_DEVICE
    return jax.devices(device)[idx] if isinstance(device, str) else device


def resolve_dtype(dtype):
    """Sanitize and check (via error raised) if argument is a dtype specification."""
    global jnp
    return jnp.dtype(dtype)


def _jaxm_to(
    x: "Array",  # noqa: F821
    device_or_dtype: Optional[Tuple[str, Any]] = None,
    device: Optional[Tuple[str, Any]] = None,
    dtype: Optional[Tuple[str, Any]] = None,
):
    """Move an array to the specified device and/or dtype."""
    global jnp, jax
    if device_or_dtype is not None:
        # user specifies only one argument - either a device or dtype
        try:
            dtype = resolve_dtype(device_or_dtype)
            return x.astype(dtype)
        except TypeError:
            device = resolve_device(device_or_dtype)
            return jax.device_put(x, device)
    else:
        # user specifies keyword arguments for device and dtype explicitly (via kwargs)
        if device is not None and dtype is not None:
            return jax.device_put(x, resolve_device(device)).astype(resolve_dtype(dtype))
        if device is not None and dtype is None:
            return jax.device_put(x, resolve_device(device)).astype(
                default_dtype_for_device(device)
            )
        elif device is None and dtype is not None:
            return x.astype(resolve_dtype(dtype))
        else:
            return x


def _tree_jaxm_to(
    x: Any,
    device_or_dtype: Optional[Tuple[str, Any]] = None,
    device: Optional[Tuple[str, Any]] = None,
    dtype: Optional[Tuple[str, Any]] = None,
):
    return jaxm.jax.tree_util.tree_map(
        lambda z: z
        if not isinstance(z, jaxm.jax.Array)
        else _jaxm_to(z, device_or_dtype, device, dtype),
        x,
    )


####################################################################################################


def _make_jax_device(platform: str, id: int):
    import jax

    return jax.devices(platform)[id]


def _pickle_device(d):
    return _make_jax_device, (d.platform, d.id)

def _make_jax_array(arr_value, device): 
    import jax
    return jax.device_put(arr_value, device=device)

def _pickle_array(arr):
    return _make_jax_array, (np.array(arr), arr.device())


def _enable_pickling_fixes():
    import jaxlib

    copyreg.pickle(jaxlib.xla_extension.Device, _pickle_device)
    copyreg.pickle(jaxlib.xla_extension.ArrayImpl, _pickle_array)


####################################################################################################


def default_dtype_for_device(device):
    """Convert a device to its default dtype (global dtype for CPU, float32 for GPU)."""
    global jnp
    device = resolve_device(device)
    if re.search("cpu", device.device_kind) is not None:
        return DEFAULT_DTYPE
    else:
        return jnp.float32


def get_default_dtype():
    """Returns the default dtype for CPU only, default dtype for GPU cannot be changed (float32)."""
    global DEFAULT_DTYPE
    return DEFAULT_DTYPE


def set_default_dtype(dtype):
    """Sets the default dtype for CPU only, default dtype for GPU cannot be changed (float32)."""
    global jnp, DEFAULT_DTYPE
    DEFAULT_DTYPE = resolve_dtype(dtype)


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
    jaxm = copy_module(jnp)

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

    def device_dtype_fn(fn, without_dtype=False, check_second_arg_for_dtype=False):
        def fn_(*args, **kw):
            device = resolve_device(kw.get("device", DEFAULT_DEVICE))
            if "device" in kw:
                del kw["device"]
            if not without_dtype:
                kw["dtype"] = kw.get("dtype", default_dtype_for_device(device))
            if check_second_arg_for_dtype and len(args) >= 2 and _is_dtype(args[1]):
                kw["dtype"] = args[1]
                args = args[:1] + args[2:]
            return jax.device_put(fn(*args, **kw), device)

        return fn_

    jaxm.array = device_dtype_fn(jnp.array, check_second_arg_for_dtype=True)
    jaxm.ones = device_dtype_fn(jnp.ones, check_second_arg_for_dtype=True)
    jaxm.zeros = device_dtype_fn(jnp.zeros, check_second_arg_for_dtype=True)
    jaxm.full = device_dtype_fn(jnp.full, check_second_arg_for_dtype=True)
    jaxm.eye = device_dtype_fn(jnp.eye)
    jaxm.arange = device_dtype_fn(jnp.arange, without_dtype=True)
    jaxm.linspace = device_dtype_fn(jnp.linspace)
    jaxm.logspace = device_dtype_fn(jnp.logspace)

    def random_fn(fn, first_arg_to_tuple=False, default_dtype=None):
        def jaxm_fn(*args, **kw):
            global key
            if "key" in kw and kw["key"] is None:
                kw = {k: v for (k, v) in kw.items() if k != "key"}
            if "key" in kw:
                key1 = kw["key"]  # use user provided key
                kw = {k: v for (k, v) in kw.items() if k != "key"}
            else:
                key1, key2 = jrandom.split(key)
                if not isinstance(key2, jax.interpreters.partial_eval.DynamicJaxprTracer):
                    key = key2
            # set correct device and dtype
            device = resolve_device(kw.get("device", DEFAULT_DEVICE))
            if "device" in kw:
                del kw["device"]
            ddtype = (
                default_dtype if default_dtype is not None else default_dtype_for_device(device)
            )
            kw["dtype"] = kw.get("dtype", ddtype)
            # make the size a tuple if this is the only positional argument
            if first_arg_to_tuple and len(args) == 1 and not hasattr(args[0], "__iter__"):
                args = [(args[0],)] + list(args)[1:]
            ret = jax.device_put(fn(key1, *args, **kw), device)
            return ret

        return jaxm_fn

    jaxm.randn = random_fn(jrandom.normal, first_arg_to_tuple=True)
    jaxm.rand = random_fn(jrandom.uniform, first_arg_to_tuple=True)
    jaxm.randint = random_fn(
        lambda key, low, high, size, dtype=None: jrandom.randint(key, size, low, high, dtype=dtype),
        default_dtype=jaxm.int64,
    )

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
    jaxm.default_dtype_for_device = default_dtype_for_device
    jaxm.make_random_keys = make_random_keys
    jaxm.to = _tree_jaxm_to

    # module bindings
    jaxm.jax = jax
    jaxm.numpy = jnp
    jaxm.lax = jax.lax
    jaxm.scipy = jsp
    jaxm.random = jrandom

    _enable_pickling_fixes()
    return jaxm


####################################################################################################
jaxm = init()
