"""Data module for Xanadue."""

from .store import DataStore, get_data_dir

__all__ = ["DataStore", "get_data_dir", "handle_correction"]


def __getattr__(name: str):
    """Lazy import to avoid circular dependency: coordinator → data → correct → coordinator."""
    if name == "handle_correction":
        from .correct import handle_correction
        return handle_correction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
