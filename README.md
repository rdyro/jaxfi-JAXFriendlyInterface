# jfi (JAX Friendly Interface) - JAX with a PyTorch-like interface

Friendly Interface to JAX.

**News: Better, improved interface! `from jfi import jaxm` is all you need!**

Creates a JAX-like module that behaves very similarly to PyTorch, so
```python
>>> from jfi import jaxm

jaxm.norm === torch.norm
jaxm.rand === torch.rand
jaxm.cat === torch.cat
jaxm.manual_seed === torch.manual_seed
```

**Make sure to import this module before anything that might import `jax` (e.g., `jaxopt`).**

```python
# DO 
from jfi import jaxm
import jaxopt

# DON'T!!!
import jaxopt
from jfi import jaxm
```

# Working with CPU and GPU

Placing arrays on GPU and CPU is easy, either specify device/dtype directly or
use `jaxm.to` to move the array to a specific device/dtype.
```python
>>> jaxm.rand(2, device="cuda")
>>> jaxm.rand(2, device="gpu", dtype=jaxm.float64)
>>> jaxm.rand(2, device="cpu")
>>> jaxm.to(jaxm.zeros(2), "cuda")
```
Arrays are created on the CPU by default, but that can be changed using
```python
jaxm.set_default_dtype(jaxm.float32) 
jaxm.set_default_device("gpu")
jaxm.get_default_device()
jaxm.get_default_dtype()
```
**Default `dtype` refers to CPU default dtype, default GPU dtype is always `float32`, but `float64` arrays can be created on the GPU by specifying the dtype explicitly or by using `jaxm.to`.**

`jaxm` behaves like numpy (jax.numpy). Some methods are
patched directly from jax.
```python
jaxm.grad === jax.grad
jaxm.jacobian === jax.jacobian
jaxm.hessian === jax.hessian
jaxm.jit === jax.jit
jaxm.vmap === jax.vmap
```

# JAX modules are accessible directly

Finally, jax-backed modules are available directly in jaxm
```python
>>> jaxm.jax
>>> jaxm.numpy
>>> jaxm.random
>>> jaxm.scipy
>>> jaxm.lax
>>> jaxm.xla
```

Additional references can be found in the source code of this module which
consists of 1 file.

# 🔪 The Sharp Bits 🔪

Random numbers are implemented using a global random key (which can also be
manually set using e.g., `jaxm.manual_seed(2023)`). However, that means parallelized
routines will generate the same random numbers.

```python
# DON'T DO THIS
jaxm.jax.vmap(lambda _: jaxm.randn(10))(jaxm.arange(10)) # every row of random numbers is the same!

# DO THIS INSTEAD
n = 10
random_keys = jaxm.make_random_keys(n)
jaxm.jax.vmap(lambda key, idx: jaxm.randn(10, key=key))(random_keys, jaxm.arange(n))
```

`jit`-ted functions will also return  the same random numbers every time
```python
# DON'T DO THIS
f = jaxm.jit(lambda x: x * jaxm.randn(3))
f(1) # [-1.12918106, -2.04245763, -0.40538156]
f(1) # [-1.12918106, -2.04245763, -0.40538156]
f(1) # [-1.12918106, -2.04245763, -0.40538156]

# DO THIS
f = jaxm.jit(lambda x, key=None: x * jaxm.randn(3, key=key))
f(1) # [-1.12918106, -2.04245763, -0.40538156]
f(1, jaxm.make_random_key()) # [-2.58426713,  0.90726101,  2.1546499 ]
# jaxm.make_random_keys(n) is also available
```

# Notes

I'm not affiliated with [JAX](https://github.com/google/jax) or
[PyTorch](https://pytorch.org/) in any way.
