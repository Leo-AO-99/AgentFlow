__version__ = "0.1.2"

# Register short-path aliases so that `from agentflow.models.xxx import ...`
# resolves to `agentflow.agentflow.models.xxx` (same for engine / tools).
# The internal package code uses these short paths throughout.
import sys as _sys
import importlib as _importlib

for _sub in ["models", "engine", "tools", "tracer", "instrumentation"]:
    _short = f"agentflow.{_sub}"
    _full  = f"agentflow.agentflow.{_sub}"
    if _short not in _sys.modules:
        try:
            _sys.modules[_short] = _importlib.import_module(_full)
        except ImportError:
            pass

del _sys, _importlib, _sub, _short, _full

from .client import AgentFlowClient, DevTaskLoader
from .config import flow_cli
from .litagent import LitAgent
from .logging import configure_logger
from .reward import reward
from .server import AgentFlowServer
from .trainer import Trainer
from .types import *
