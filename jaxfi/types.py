from __future__ import annotations

try:
    from jax import Array # noqa F401
except ImportError:
    from jaxlib.xla_extension import DeviceArray as Array # noqa F401
