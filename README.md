# `jaxfi` (JAX Friendly Interface) - JAX with a PyTorch-like interface

Friendly Interface to JAX, that behaves similar to PyTorch while maintaining compatibility.

- [`jaxfi` (JAX Friendly Interface) - JAX with a PyTorch-like interface](#jaxfi-jax-friendly-interface---jax-with-a-pytorch-like-interface)
- [Working with CPU and GPU](#working-with-cpu-and-gpu)
- [JAX modules are accessible directly](#jax-modules-are-accessible-directly)
- [🔪 The Sharp Bits 🔪](#-the-sharp-bits-)
- [Notes](#notes)
- [Installation](#installation)
- [Changelog](#changelog)

**News: Better, improved interface! `import jaxfi as jaxm` is all you need!**

Creates a JAX-like module that behaves very similarly to PyTorch, so
```python
>>> import jaxfi as jaxm

jaxm.norm === torch.norm
jaxm.rand === torch.rand
jaxm.cat === torch.cat
jaxm.manual_seed === torch.manual_seed
```


**Make sure to import this module before anything that might import `jax` (e.g., `jaxopt`).**

```python
# DO 
import jaxfi as jaxm
import jaxopt

# DON'T!!!
import jaxopt
import jaxfi as jaxm
```

# Working with CPU and GPU

> JAX has automatic device placement in functions, so omit the `device` argument
> when creating arrays in functions, i.e., in functions, specify only the dtype.

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
**Default `dtype` refers to CPU default dtype, default GPU dtype is always `float32`, but `float64` arrays can be created on the GPU by specifying the `dtype` explicitly or by using `jaxm.to`.**

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
```

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

# Installation

```bash
$ pip install jaxfi
```

The package name recently change from `jfi` to `jaxfi`, PyPI hosts it as `jaxfi`.

Alternatively, to install from source, issue
```bash
$ pip install .
```
from the project root, or simply run
```bash
$ pip install git+https://github.com/rdyro/jaxfi-JAXFriendlyInterface.git
```

If you wish to let JAX (not `jaxfi`) work alongside PyTorch in the same virtual
environment, set/export the environment variable `JAXFI_LOAD_SYSTEM_CUDA_LIBS=true`
before importing `jaxfi` or `jax` for the first time.
```bash
$ echo 'export JAXFI_LOAD_SYSTEM_CUDA_LIBS=true' >> ~/.bashrc
$ echo 'export JAXFI_LOAD_SYSTEM_CUDA_LIBS=true' >> ~/.zshrc
```
This will instruct `jaxfi` to dynamically load the system CUDA libraries.


# Changelog

- version 0.7.3
  - fixed random functions not accepting `key=` kwargs for under-jit random number generation

- version 0.7.0
  - `jaxfi` is now identical with `jaxm` so that both `import jaxfi as jaxm` and `from jaxfi import jaxm` work
  - this change helps (at least the VSCode) Pylance resolve member fields in `jaxfi`

- version 0.6.6
  - random functions now (correctly) produce uncommitted arrays (see [https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices](https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices))
  - added a PyTorch-like randperm function (implemented as argsort(rand))

- version 0.6.5
  - added the ability to dynamically load the system CUDA libraries so allowing
  JAX to live in harmony with PyTorch, set the environment variable
  `JAXFI_LOAD_SYSTEM_CUDA_LIBS=true` to enable this feature

- version 0.6.3
  - `jaxm.to` now also moves numpy, not just jax, arrays to a device and dtype
  - experimental `auto_pmap` function available, automatically assigning first
    batch dimension to multiple devices, e.g., dividing 16 tasks into 6 CPUs

- version 0.6.0
  - official name change from `jfi` to `jaxfi`

- version 0.5.0
    - settled on the default numpy module copy behavior
    - omit `device` when creating arrays in functions - this now works correctly
    - introduced more tests
