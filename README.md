# jfi - JAXFriendlyInterface
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
```

Finally, jax-backed modules are available directly in jaxm
```python
>>> jaxm.jax
>>> jaxm.numpy
>>> jaxm.random
>>> jaxm.scipy
>>> jaxm.lax
>>> jaxm.xla
```

Additional reference can be found in the source code of this module which
consists of 1 file.
