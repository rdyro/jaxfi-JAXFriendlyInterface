from .api import jaxm
from .utils import resolve_device, resolve_dtype, default_dtype_for_device
from .utils import get_default_device, get_default_dtype, set_default_dtype, set_default_device
from .experimental.jit import autojit, nestedautojit