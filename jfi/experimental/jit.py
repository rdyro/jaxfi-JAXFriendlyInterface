import inspect
from typing import Callable

from jax import Array, jit
from jax.tree_util import tree_flatten, tree_unflatten, tree_map


class AutoJIT:
    def __init__(self, fn: Callable):
        self.fn = fn
        self.compiled_fn = None
        self.fn_signature = inspect.signature(self.fn)

    def __call__(self, *args, **kw):
        # we apply defaults to the arguments using the function signature
        args_kw = self.fn_signature.bind(*args, **kw)
        args_kw.apply_defaults()
        args, kw = args_kw.args, args_kw.kwargs

        # if this is the first time we see the arguments to the function
        # we're going to determine which arguments are `static`
        if self.compiled_fn is None:
            # 1. for every variable argument, we check if it's a jax.Array, if not, it's static
            static_argnums = [i for (i, arg) in enumerate(args) if not isinstance(arg, Array)]
            # 2. similarly, for every keyword argument, we check and set static if not jax.Array
            static_argnames = [k for (k, v) in kw.items() if not isinstance(v, Array)]
            # 3. we jit the function with the static arguments and argnames
            self.compiled_fn = jit(
                    self.fn, static_argnums=static_argnums, static_argnames=static_argnames
            )
        # we just call the compiled function
        return self.compiled_fn(*args, **kw)

autojit = AutoJIT # alias


class NestedAutoJIT:
    """The advanced version of AutoJIT, which flattens the arguments to handle pytree inputs."""
    def __init__(self, fn: Callable):
        self.fn = fn
        # we now have a cache of compiled functions for each combination of provided arguments
        self.compiled_fn_cache = dict() 
        self.fn_signature = inspect.signature(self.fn)

    def __call__(self, *args, **kw):
        # we apply defaults to the arguments using the function signature
        args_kw = self.fn_signature.bind(*args, **kw)
        args_kw.apply_defaults()
        args, kw = args_kw.args, args_kw.kwargs

        # 1. first, we flatten the arguments and keyword arguments
        flat_args, args_struct = tree_flatten(args)
        flat_kw, kw_struct = tree_flatten(kw)
        args_types, kw_types = tree_map(type, flat_args), tree_map(type, flat_kw)
        # 2. we produce a "unique" identifier for provided arguments: the structure and types
        cache_key = (args_struct, kw_struct, tuple(args_types), tuple(kw_types))
        # 3. underneath, we're going to call the function using with all arguments flattened
        flat_args_kw = tuple(flat_args) + tuple(flat_kw)
        if cache_key not in self.compiled_fn_cache:

            # 3. we produce a flat argument version of the function
            def flat_fn(*flat_args_kw):
                # 4. we need to unflatten the arguments and keyword arguments
                args, kw_args = flat_args_kw[: len(flat_args)], flat_args_kw[len(flat_args) :]
                args = tree_unflatten(args_struct, args)
                kw = tree_unflatten(kw_struct, kw_args)
                # 5. we call the function with the unflattened arguments
                return self.fn(*args, **kw)

            # 6. we can now determine, using the flat argument version, which args need to be static
            static_argnums = [
                i for (i, arg) in enumerate(flat_args_kw) if not isinstance(arg, Array)
            ]
            # 7. we can now compile the flat argument version of the function
            self.compiled_fn_cache[cache_key] = jit(flat_fn, static_argnums=static_argnums)

        # 8. we can now call the compiled function with the flat arguments
        return self.compiled_fn_cache[cache_key](*flat_args)

nestedautojit = NestedAutoJIT # alias