from __future__ import annotations

from functools import partial
from typing import Callable

import jax
from jax.core import Primitive
from jax.interpreters import xla, mlir
from jax import tree

ID = 0


def device_select(cpu_fn: Callable, gpu_fn: Callable) -> Callable:
    global ID
    prim = Primitive(f"device_select_{ID}")
    ID += 1

    prim.multiple_results = True
    prim.def_impl(partial(xla.apply_primitive, prim))

    output_struct = None

    def cpu_flat_fn(*args, **kw):
        nonlocal output_struct
        out = cpu_fn(*args, **kw)
        leaves, struct = tree.flatten(out)
        if output_struct is None:
            output_struct = struct
        return leaves

    def gpu_flat_fn(*args, **kw):
        nonlocal output_struct
        out = gpu_fn(*args, **kw)
        leaves, struct = tree.flatten(out)
        if output_struct is None:
            output_struct = struct
        return leaves

    def abstract_eval_fn(*args, **kw):
        leaves_cpu = jax.make_jaxpr(cpu_flat_fn)(*args, **kw).out_avals
        leaves_gpu = jax.make_jaxpr(gpu_flat_fn)(*args, **kw).out_avals
        msg = (
            "Mismatch between CPU and GPU outputs. Flattened outputs:\n"
            + str(leaves_cpu)
            + "\n"
            + str(leaves_gpu)
        )
        msg += "\nProvide two functions with the same signature and return types."
        assert len(leaves_cpu) == len(leaves_gpu) and all(
            x.shape == y.shape for x, y in zip(leaves_cpu, leaves_gpu)
        ), msg
        return leaves_cpu

    prim.def_abstract_eval(abstract_eval_fn)

    def cpu_fn_(*args, **kw):
        return mlir.lower_fun(cpu_flat_fn, multiple_results=True)(*args, **kw)

    def gpu_fn_(*args, **kw):
        return mlir.lower_fun(gpu_flat_fn, multiple_results=True)(*args, **kw)

    mlir.register_lowering(prim, cpu_fn_, platform="cpu")
    mlir.register_lowering(prim, gpu_fn_, platform="gpu")

    def forward_fn(*args):
        nonlocal output_struct
        out_flat = prim.bind(*args)
        return tree.unflatten(output_struct, out_flat)

    return forward_fn
