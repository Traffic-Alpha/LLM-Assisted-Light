"""Lightweight core exports for the refactored TSC controller."""

from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from llm_tsc.agent import ConversationMessage, SimplifiedReActAgent
    from llm_tsc.config import Config, load_config
    from llm_tsc.maxpressure import MaxPressureController, choose_maxpressure_phase
    from llm_tsc.prompts import PromptManager
    from llm_tsc.tools_adapter import TSCToolsAdapter
    from llm_tsc.tools_registry import ToolRegistry, tool


_EXPORTS = {
    "SimplifiedReActAgent": ("llm_tsc.agent", "SimplifiedReActAgent"),
    "ConversationMessage": ("llm_tsc.agent", "ConversationMessage"),
    "ToolRegistry": ("llm_tsc.tools_registry", "ToolRegistry"),
    "tool": ("llm_tsc.tools_registry", "tool"),
    "PromptManager": ("llm_tsc.prompts", "PromptManager"),
    "TSCToolsAdapter": ("llm_tsc.tools_adapter", "TSCToolsAdapter"),
    "Config": ("llm_tsc.config", "Config"),
    "load_config": ("llm_tsc.config", "load_config"),
    "MaxPressureController": ("llm_tsc.maxpressure", "MaxPressureController"),
    "choose_maxpressure_phase": ("llm_tsc.maxpressure", "choose_maxpressure_phase"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'llm_tsc' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
