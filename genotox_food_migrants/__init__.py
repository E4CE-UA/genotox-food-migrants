"""Unambiguous development import for the canonical modules in src/."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[1] / "src")]
__version__ = "0.2.0"
