# jfi - JAXFriendlyInterface
Friendly Interface to JAX.

**News: Better, improved interface! `from jfi import jaxm` is all you need!**

Creates a JAX-like module that behaves very similarly to PyTorch, so
```
>>> from jfi import jaxm

jaxm.norm === torch.norm
jaxm.rand === torch.rand
jaxm.cat === torch.cat
jaxm.manual_seed === torch.manual_seed
```

However, `jaxm` otherwise behaves like numpy (jax.numpy). Some methods are
patched directly from jax.
```
jaxm.grad === jax.grad
jaxm.jacobian === jax.jacobian
jaxm.hessian === jax.hessian
jaxm.jit === jax.jit
```

Finally, jax-backed modules are available directly in jfi
```
>>> jaxm.jax
>>> jaxm.numpy
>>> jaxm.random
>>> jaxm.scipy
>>> jaxm.lax
>>> jaxm.xla
```

Additional reference can be found in the source code of this module which
consists of 1 file and 90 lines of code.
