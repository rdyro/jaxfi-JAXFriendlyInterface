import re
from typing import Tuple, Any, Optional
import os
import copyreg

import numpy as np

from . import globals

ModuleType = type(os)  # a Python version agnostic to the get the module type


def make_random_keys(n):
    keys = globals.jrandom.split(globals.key, n + 1)
    globals.key = keys[-1]
    return keys[:-1]


def make_random_key():
    return make_random_keys(1)[0]


def copy_module(mod, recursive=True):
    new_mod = ModuleType(mod.__name__ + "_jaxfi")
    for attr in dir(mod):
        if not attr.startswith("_"):
            if isinstance(getattr(mod, attr), ModuleType) and recursive:
                setattr(new_mod, attr, copy_module(getattr(mod, attr), recursive=recursive))
            else:
                setattr(new_mod, attr, getattr(mod, attr))
    return new_mod


####################################################################################################


def _is_dtype(x):
    try:
        globals.jnp.dtype(x)
        return True
    except TypeError:
        return False


def resolve_device(device: Tuple[str, Any, None], idx: int = 0):
    """Convert device name to the device handle."""
    if device is None:
        #return globals.DEFAULT_DEVICE
        return None
    return globals.jax.devices(device)[idx] if isinstance(device, str) else device


def resolve_dtype(dtype):
    """Sanitize and check (via error raised) if argument is a dtype specification."""
    return globals.jnp.dtype(dtype)


def _jaxm_to(
    x: "Array",  # noqa: F821
    device_or_dtype: Optional[Tuple[str, Any]] = None,
    device: Optional[Tuple[str, Any]] = None,
    dtype: Optional[Tuple[str, Any]] = None,
):
    """Move an array to the specified device and/or dtype."""
    if device_or_dtype is not None:
        # user specifies only one argument - either a device or dtype
        try:
            dtype = resolve_dtype(device_or_dtype)
            return x.astype(dtype)
        except TypeError:
            device = resolve_device(device_or_dtype)
            return globals.jax.device_put(x, device)
    else:
        # user specifies keyword arguments for device and dtype explicitly (via kwargs)
        if device is not None and dtype is not None:
            return globals.jax.device_put(x, resolve_device(device)).astype(resolve_dtype(dtype))
        if device is not None and dtype is None:
            return globals.jax.device_put(x, resolve_device(device)).astype(
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
    return globals.jax.tree_util.tree_map(
        lambda z: z
        if not isinstance(z, globals.jax.Array)
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
    device = resolve_device(device)
    if device is None or re.search("cpu", device.device_kind) is not None:
        return globals.DEFAULT_DTYPE
    else:
        return globals.jnp.float32


def get_default_dtype():
    """Returns the default dtype for CPU only, default dtype for GPU cannot be changed (float32)."""
    return globals.DEFAULT_DTYPE


def set_default_dtype(dtype):
    """Sets the default dtype for CPU only, default dtype for GPU cannot be changed (float32)."""
    globals.DEFAULT_DTYPE = resolve_dtype(dtype)


def get_default_device():
    """Returns the default device for which to create arrays on."""
    return globals.DEFAULT_DEVICE


def set_default_device(device):
    """Sets the default device for which to create arrays on."""
    globals.DEFAULT_DEVICE = resolve_device(device)
    globals.jax.config.update("jax_default_device", globals.DEFAULT_DEVICE)


def manual_seed(val):
    """Sets the random seed for the random number generator."""
    assert globals.jrandom is not None
    globals.key = globals.jrandom.PRNGKey(val)
