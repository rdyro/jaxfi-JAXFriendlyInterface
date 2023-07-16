import sys
from pathlib import Path

path = Path(__file__).absolute().parents[1]
if str(path) not in sys.path:
    sys.path.insert(path, 0)


from jfi import jaxm

try:
    jaxm.resolve_device("cuda")
    HAS_GPU = True
except RuntimeError:
    HAS_GPU = False

def test_optax():
    try:
        import optax
    except ImportError:
        print("Module `optax` not found, skipping test")
        return

    rand_fn = lambda dtype, device: jaxm.randn(3, dtype=dtype, device=device)
    devices = [jaxm.resolve_device(x) for x in (["cpu", "cuda"] if HAS_GPU else ["cpu"])]
    for device in devices:
        for dtype in [jaxm.float32, jaxm.float64]:
            param = rand_fn(dtype, device)
            opt = optax.adam(1e-3)
            opt_state = opt.init([param])
            gs = rand_fn(dtype, device)
            updates, opt_state = opt.update([gs], opt_state, [param])
            param = optax.apply_updates([param], updates)
            print(param)