import inspect
from typing import Callable, Iterable

import numpy as np

from jax import Array, jit
from jax.tree_util import tree_flatten, tree_unflatten, tree_map


_jit_types = (Array, float, np.ndarray)
_iffy_jit_types = (int,)


class AutoJIT:
    """The AutoJIT class, which automatically compiles a function on first call,
    statically labeling non-JAX-representable arguments and keywords (e.g.,
    strings)."""

    def __init__(
        self,
        fn: Callable,
        jit_ints: bool = False,
        static_argnums: Iterable = None,
        static_argnames: Iterable = None,
    ):
        self.fn = fn
        self.compiled_fn = None
        self.fn_signature = inspect.signature(self.fn)
        self.static_argnums = set(static_argnums) if static_argnums is not None else set()
        self.static_argnames = set(static_argnames) if static_argnames is not None else set()
        if jit_ints:
            self._jit_types = _jit_types + _iffy_jit_types
        else:
            self._jit_types = _jit_types

    def __call__(self, *args, **kw):
        args_kw = self.fn_signature.bind(*args, **kw)
        args_kw.apply_defaults()
        args, kw = args_kw.args, args_kw.kwargs

        if self.compiled_fn is None:
            static_argnums = set(
                i for (i, arg) in enumerate(args) if not isinstance(arg, self._jit_types)
            )
            static_argnames = set(k for (k, v) in kw.items() if not isinstance(v, self._jit_types))

            static_argnums = static_argnums.union(self.static_argnums)
            static_argnames = static_argnames.union(self.static_argnames)

            self.compiled_fn = jit(
                self.fn,
                static_argnums=tuple(static_argnums),
                static_argnames=tuple(static_argnames),
            )
        return self.compiled_fn(*args, **kw)


autojit = AutoJIT  # alias


class NestedAutoJIT:
    """The nested version of AutoJIT, which flattens the arguments to handle pytree inputs."""

    def __init__(self, fn: Callable, jit_ints: bool = False):
        self.fn = fn
        self.compiled_fn_cache = dict()
        self.fn_signature = inspect.signature(self.fn)
        if jit_ints:
            self._jit_types = _jit_types + _iffy_jit_types
        else:
            self._jit_types = _jit_types

    def __call__(self, *args, **kw):
        args_kw = self.fn_signature.bind(*args, **kw)
        args_kw.apply_defaults()
        args, kw = args_kw.args, args_kw.kwargs

        flat_args, args_struct = tree_flatten(args)
        flat_kw, kw_struct = tree_flatten(kw)
        args_types, kw_types = tree_map(type, flat_args), tree_map(type, flat_kw)

        cache_key = (args_struct, kw_struct, tuple(args_types), tuple(kw_types))

        flat_args_kw = tuple(flat_args) + tuple(flat_kw)
        if cache_key not in self.compiled_fn_cache:

            def flat_fn(*flat_args_kw):
                args, kw_args = flat_args_kw[: len(flat_args)], flat_args_kw[len(flat_args) :]
                args = tree_unflatten(args_struct, args)
                kw = tree_unflatten(kw_struct, kw_args)
                return self.fn(*args, **kw)

            static_argnums = [
                i for (i, arg) in enumerate(flat_args_kw) if not isinstance(arg, self._jit_types)
            ]
            self.compiled_fn_cache[cache_key] = jit(flat_fn, static_argnums=static_argnums)

        return self.compiled_fn_cache[cache_key](*flat_args_kw)


nestedautojit = NestedAutoJIT  # alias
