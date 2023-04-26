from typing import Callable, Optional, Optional

import torch
from torch import Tensor
import numpy as np

from .. import jaxm, resolve_device

from jax import ShapeDtypeStruct, tree_util as tu, pure_callback, Array

torch_dtype_map = {
    None: torch.float32,
    jaxm.float32: torch.float32,
    jaxm.float64: torch.float64,
    jaxm.int32: torch.int32,
    jaxm.int64: torch.int64,
}


def torch_device_map(device: "DeviceT") -> torch.device:
    """Convert a JAX device to PyTorch device.

    Args:
        device (DeviceT): JAX device.

    Raises:
        ValueError: Raises error if device is not recognized by JAX or is neither CPU nor GPU.

    Returns:
        torch.device: The device under torch.
    """
    device = resolve_device(device) if isinstance(device, str) else device
    if device is None or device.platform == "cpu":
        return torch.device("cpu")
    elif device.platform == "gpu":
        return torch.device(f"cuda:{device.id}")
    else:
        raise ValueError(f"Unknown device: {device}")


def map_to_torch(x: np.ndarray, dtype: "DtypeT" = None, device: "DeviceT" = None) -> Tensor:
    """Convert a numpy array to a torch tensor.

    Args:
        x (np.ndarray): Numpy array.
        dtype (DtypeT, optional): JAX dtype to map to. Defaults to None.
        device (DeviceT, optional): JAX device to map to. Defaults to None.

    Returns:
        Tensor: Tensor on the corresponding torch device and dtype.
    """
    if isinstance(x, np.ndarray):
        dtype = dtype if x.dtype in [np.float32, np.float64] else x.dtype
        dtype = torch_dtype_map[dtype]
        return torch.as_tensor(x, dtype=dtype, device=torch_device_map(device))
    else:
        return x


def map_to_numpy(x: Tensor, dtype: "DtypeT" = None) -> np.ndarray:
    """Convert a torch tensor to a numpy array.

    Args:
        x (Tensor): Torch tensor
        dtype (DtypeT, optional): JAX dtype. Defaults to None.

    Returns:
        np.ndarray: Numpy array with a corresponding dtype on CPU.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype)
    else:
        return x


def torch2jax(x: Tensor, dtype=None, device=None) -> Array:
    """Convert a torch tensor to a JAX array.

    Args:
        x (Tensor): Torch tensor.
        dtype (_type_, optional): JAX dtype. Defaults to None.
        device (_type_, optional): JAX device. Defaults to None.

    Returns:
        Array: JAX array.
    """
    return jaxm.array(map_to_numpy(x), device=device, dtype=dtype)


def wrap_torch_fn(
    fn: Callable,
    outshape_fn: Optional[Callable] = None,
    shape: Optional[tuple] = None,
    device: "DeviceT" = None,
    dtype: "DtypeT" = None,
) -> Callable:
    """Wrap a torch function to a JAX function using pure_callback with correct
    dtype and device placement.

    Args:
        fn (Callable): Function in Torch.
        outshape_fn (Optional[Callable], optional): Function that evaluates
                                                    output shape given an input shape.
        shape (Optional[tuple], optional): Output shape.
        device (DeviceT, optional): JAX device. Defaults to None.
        dtype (DtypeT, optional): JAX dtype. Defaults to None.

    Returns:
        Callable: A JAX converted function.
    """
    assert outshape_fn is not None or shape is not None

    def _wrapped_fn(*args):
        args = tu.tree_map(lambda x: map_to_torch(x, dtype=dtype, device=device), args)
        out = fn(*args)
        out = tu.tree_map(lambda x: map_to_numpy(x, dtype=dtype), out)
        return out

    def wrapped_fn(*args):
        s = shape if shape is not None else outshape_fn(*args)
        return jaxm.to(pure_callback(_wrapped_fn, s, *args), device=device, dtype=dtype)

    return wrapped_fn


def create_custom_jvp(
    torch_fn: Callable, *args, dtype: "DtypeT" = None, device: "DeviceT" = None
) -> Callable:
    """Converts a Torch function to a JAX function with a custom_jvp.

    Args:
        torch_fn (Callable): Function in Torch.
        dtype (DtypeT, optional): JAX dtype. Defaults to None.
        device (DeviceT, optional): JAX device. Defaults to None.

    Returns:
        Callable: A JAX function with a defined custom_jvp.
    """
    jdtype = jaxm.dtype(dtype)
    outputs = torch_fn(*args)
    output_shape = tu.tree_map(lambda x: ShapeDtypeStruct(x.shape, dtype=jdtype), outputs)

    def fn(*args):
        return wrap_torch_fn(torch_fn, shape=output_shape, dtype=dtype, device=device)(*args)

    if depth <= 0:
        return fn

    def f_jvp_torch(primals, tangents):
        return torch.func.jvp(torch_fn, primals, tangents)

    def f_jvp(primals, tangents):
        primals_out = fn(*primals)
        tangets_out = wrap_torch_fn(f_jvp_torch, shape=output_shape, dtype=dtype, device=device)(
            primals, tangents
        )
        return primals_out, tangets_out

    fn = jaxm.jax.custom_jvp(fn)
    fn.defjvp(f_jvp)
    return fn


def create_custom_vjp(
    torch_fn: Callable,
    *args,
    dtype: "DtypeT" = None,
    device: "DeviceT" = None,
    depth: int = 1,
    create_jvp: bool = False,
) -> Callable:
    """_summary_

    Args:
        torch_fn (Callable): Function in Torch.
        dtype (DtypeT, optional): JAX dtype. Defaults to None.
        device (DeviceT, optional): JAX device. Defaults to None.
        depth (int, optional): Number of recursive backwards derivatives to define. Defaults to 1.
        create_jvp (bool, optional): Whether to also create a jvp; not supported currently.

    Raises:
        ValueError: Raises an error if user requests a jvp because JAX doesn't currently support it.

    Returns:
        Callable: A JAX function with a (recursively) defined custom_vjp.
    """

    jdtype = jaxm.dtype(dtype)
    outputs = torch_fn(*args)
    output_shape = tu.tree_map(lambda x: ShapeDtypeStruct(x.shape, dtype=jdtype), outputs)

    def fn(*args):
        return wrap_torch_fn(torch_fn, shape=output_shape, dtype=dtype, device=device)(*args)

    if depth <= 0:
        return fn

    fn = jaxm.jax.custom_vjp(fn)

    def fwd_fn(*args):
        return fn(*args), args

    def bwd_fn_torch(args, gs):
        _, vjp_fn = torch.func.vjp(torch_fn, *args)
        return vjp_fn(gs)

    if create_jvp:
        raise ValueError("This is currently not supported by JAX")
        bwd_fn = create_custom_jvp(bwd_fn_torch, args, outputs, dtype=dtype, device=device, depth=1)
    else:
        bwd_fn = create_custom_vjp(
            bwd_fn_torch, args, outputs, dtype=dtype, device=device, depth=depth - 1
        )
        fn.defvjp(fwd_fn, bwd_fn)

    return fn
