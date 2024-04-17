import time
import sys
import os
import pdb

os.environ["XLA_FLAGS"] = f"--xla_backend_optimization_level={sys.argv[1]}"

from jaxfi import jaxm # noqa: E402


n = 1000

def fn(x):
    for i in range(n):
        x = (x * x) / jaxm.norm(x)
    return x

x = jaxm.randn(10, device="cpu", dtype=jaxm.float64)
t = time.time()
fn(x)
t = time.time() - t
print(f"Eager takes {t:.4e} s")

os.environ["XLA_FLAGS"] = "--xla_backend_optimization_level=0"

t = time.time()
fn_jit = jaxm.jit(fn)
fn_jit(x)
t = time.time() - t
print(f"Compilation + exec takes {t:.4e} s")

t = time.time()
for i in range(100):
    fn_jit(x).block_until_ready()
t = (time.time() - t) / 100
print(f"Exec takes {t:.4e} s")

pdb.set_trace()
