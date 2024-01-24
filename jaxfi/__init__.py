try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from importlib_metadata import version
__version__ = version(__name__)

from .api import jaxm  # noqa: F401h
from .utils import resolve_device, resolve_dtype, default_dtype_for_device  # noqa: F401
from .utils import get_default_device, get_default_dtype  # noqa: F401
from .utils import set_default_dtype, set_default_device  # noqa: F401
from .experimental.jit import autojit, nestedautojit  # noqa: F401
