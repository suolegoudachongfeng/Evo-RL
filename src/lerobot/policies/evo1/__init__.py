from .configuration_evo1 import Evo1Config

__all__ = ["Evo1Config", "EVO1Policy", "make_evo1_pre_post_processors"]


def __getattr__(name: str):
    if name == "EVO1Policy":
        from .modeling_evo1 import EVO1Policy

        return EVO1Policy
    if name == "make_evo1_pre_post_processors":
        from .processor_evo1 import make_evo1_pre_post_processors

        return make_evo1_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
