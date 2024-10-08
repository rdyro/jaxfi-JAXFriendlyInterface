import re
from typing import Tuple, Any, Optional
import os
import copyreg

import numpy as np

from . import globals
from .types import Array

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


def resolve_device(device: Tuple[str, Any, None], idx: int = -1):
    """Convert device name to the device handle."""
    if device is None:
        return None
    if not isinstance(device, str): 
        return device # already a device (or sharding)
    if isinstance(device, str) and ":" in device:
        assert idx == -1, "When using the ':' syntax, do not specify idx separately."
        device, idx = device.split(":", 1)
        idx = int(idx)
    idx = 0 if idx == -1 else idx
    devices_list = globals.jax.devices(device)
    assert 0 <= idx < len(devices_list), f"Invalid device index {idx} for device {device}. Available devices: {devices_list}"
    return devices_list[idx]


def resolve_dtype(dtype):
    """Sanitize and check (via error raised) if argument is a dtype specification."""
    return globals.jnp.dtype(dtype)


def _jaxm_to(
    x: Array,  # noqa: F821
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
            ret = x.astype(resolve_dtype(dtype))
            return ret
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
        # inspired by https://github.com/patrick-kidger/equinox
        if not isinstance(z, (Array, np.ndarray, np.generic)) 
        else _jaxm_to(z, device_or_dtype, device=device, dtype=dtype),
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


def _pickle_array(arr: Array):
    devices = tuple(arr.devices())
    if len(devices) > 1:
        raise NotImplementedError("jaxfi does not support pickling sharded arrays")
    return _make_jax_array, (np.array(arr), devices[0])


def _enable_pickling_fixes():
    import jaxlib

    copyreg.pickle(jaxlib.xla_extension.Device, _pickle_device)
    try:
        copyreg.pickle(jaxlib.xla_extension.ArrayImpl, _pickle_array)
    except AttributeError:
        copyreg.pickle(jaxlib.xla_extension.DeviceArray, _pickle_array)


####################################################################################################


def default_dtype_for_device(device):
    """Convert a device to its default dtype (global dtype for CPU, float32 for GPU)."""
    device = resolve_device(device)
    if device is not None:
        # this might be a sharding
        if not hasattr(device, "device_kind"):
            try:
                device_kind = list(device.device_set)[0].device_kind
            except: # noqa: E722
                device_kind = None # could not determine device kind
        else:
            device_kind = device.device_kind
    else:
        device_kind = None
    if device is None or (device_kind is not None and re.search("cpu", device_kind) is not None):
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


def set_default_device(device, idx=0):
    """Sets the default device for which to create arrays on."""
    if isinstance(device, str) and ":" in device:
        device, idx = device.split(":", 1)
        idx = int(idx)
    globals.DEFAULT_DEVICE = resolve_device(device, idx)
    try:
        globals.jax.config.update("jax_default_device", globals.DEFAULT_DEVICE)
    except AttributeError:
        pass


def manual_seed(val):
    """Sets the random seed for the random number generator."""
    assert globals.jrandom is not None
    globals.key = globals.jrandom.PRNGKey(val)
