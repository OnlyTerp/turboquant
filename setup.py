"""Compatibility shim.

All real packaging metadata now lives in ``pyproject.toml`` (PEP 621).
This file is kept so older ``pip install -e .`` workflows on environments
that don't yet prefer PEP 660 editable installs still work, but it is
effectively a no-op: setuptools reads everything from ``pyproject.toml``.
"""

from setuptools import setup

setup()
