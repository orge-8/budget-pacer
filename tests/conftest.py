"""pytest 共享夹具：按文件路径加载插件模块，避免不同插件的同名 plugin 互相覆盖。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_plugin_module():
    """以独立模块名加载插件，规避扁平 sys.path 下的同名冲突。

    必须先把模块登记进 sys.modules：插件用了 `from __future__ import annotations`，
    dataclass 解析字符串注解时要能回查到模块本身，否则会抛 AttributeError。
    """
    spec = importlib.util.spec_from_file_location("budget_pacer_plugin", ROOT / "plugin.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 plugin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


@pytest.fixture(scope="session")
def bp():
    """插件模块，提供纯函数与配置模型的访问入口。"""
    return load_plugin_module()
