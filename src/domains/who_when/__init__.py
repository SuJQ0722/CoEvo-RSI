"""Load the plugin on demand so ``python -m ...runtime.evaluation`` stays clean."""

__all__ = ["PLUGIN", "WhoWhenPlugin"]


def __getattr__(name):
    if name in __all__:
        from .plugin import PLUGIN, WhoWhenPlugin

        globals().update(PLUGIN=PLUGIN, WhoWhenPlugin=WhoWhenPlugin)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
