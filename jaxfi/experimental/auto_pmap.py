from ..globals import jaxm

import jax
from jax.tree_util import tree_flatten, tree_unflatten
from jax import api_util
from jax import pmap, vmap
from jax import numpy as jnp


def auto_pmap(fn, device="cpu", in_axes=0, out_axes=0):
    assert device.lower() in {"cpu"}
    devices = jax.devices(device)
    n_devices = len(devices)

    def _pmap_fn(*args):
        args_flat, args_struct = tree_flatten(args)
        # in_axes_flat = tree_flatten(in_axes)[0]
        in_axes_flat = api_util.flatten_axes("hello", args_struct, in_axes)
        assert all(
            axes in (0, None) for axes in in_axes_flat
        ), "We only support 0 or None in in_axes"
        assert len(args_flat) == len(in_axes_flat)

        batch_sizes = [x.shape[0] for x, axes in zip(args_flat, in_axes_flat) if axes == 0]
        batch_size = batch_sizes[0]
        assert all(x == batch_size for x in batch_sizes), "All batch sizes must be equal"
        n_devices_ = min(n_devices, batch_size)

        floor_size = (batch_size // n_devices_) * n_devices_
        print(f"batch_size: {batch_size}, floor_size: {floor_size}")

        # pmap #####################################################################################
        args_flat_pmap = [
            x if axes is None else x[:floor_size, ...].reshape((n_devices_, -1) + x.shape[1:])
            for x, axes in zip(args_flat, in_axes_flat)
        ]
        out_pmap = pmap(
            vmap(fn, in_axes=in_axes),
            devices=devices[:n_devices_],
            in_axes=in_axes,
        )(*tree_unflatten(args_struct, args_flat_pmap))
        out_pmap_flat, out_struct = tree_flatten(out_pmap)
        out_pmap_flat = [x.reshape((floor_size,) + x.shape[2:]) for x in out_pmap_flat]
        # pmap #####################################################################################

        # pmap2 ####################################################################################
        if floor_size < batch_size:
            remain_size = batch_size - floor_size
            args_flat_pmap2 = [
                x if axes is None else x[floor_size:, ...].reshape((remain_size, 1) + x.shape[1:])
                for x, axes in zip(args_flat, in_axes_flat)
            ]
            out_pmap2 = pmap(
                vmap(fn, in_axes=in_axes),
                devices=devices[:remain_size],
                in_axes=in_axes,
            )(*tree_unflatten(args_struct, args_flat_pmap2))
            out_pmap_flat2 = tree_flatten(out_pmap2)[0]
            out_pmap_flat2 = [x.reshape((remain_size,) + x.shape[2:]) for x in out_pmap_flat2]
            out_all_flat = [
                jnp.concatenate((x, y), axis=0) for x, y in zip(out_pmap_flat, out_pmap_flat2)
            ]
            ret = tree_unflatten(out_struct, out_all_flat)
        else:
            ret = tree_unflatten(out_struct, out_pmap_flat)

        return jaxm.to(ret, device=device)
        # pmap2 ####################################################################################

    return _pmap_fn
