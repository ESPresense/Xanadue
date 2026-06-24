"""Data module for Xanadue."""

from .store import DataStore, get_data_dir
from .correct import handle_correction

__all__ = ["DataStore", "get_data_dir", "handle_correction"]
