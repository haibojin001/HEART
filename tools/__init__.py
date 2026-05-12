from . import finance, math_tools, web_apis

from ..toolface import ToolFace


def register_all(tf: ToolFace) -> None:
    finance.register(tf)
    math_tools.register(tf)
    web_apis.register(tf)


__all__ = ["finance", "math_tools", "web_apis", "register_all"]
