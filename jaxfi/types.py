from __future__ import annotations

try:
    from jax import Array
except ImportError:
    from jaxlib.xla_extension import DeviceArray as Array
