from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_CONFIG_PATH = Path(__file__).with_name("config.py")
_MODULE_NAME = "_phase2_cross_modality_alignment_config"


def _load_local_config_module():
    module = sys.modules.get(_MODULE_NAME)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Phase 2 config from {_CONFIG_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_CONFIG_MODULE = _load_local_config_module()

for _name in dir(_CONFIG_MODULE):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_CONFIG_MODULE, _name)

__all__ = [name for name in globals() if not name.startswith("_")]