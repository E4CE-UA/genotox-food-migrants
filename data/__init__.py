"""Data package.

This file exists so that the ``data/`` directory ships as an importable
package inside the wheel. Nothing imports from it: the notebook reads the
files by path through ``src.paths.DATA``. Making it a package is what lets
``pip install git+.../genotox-food-migrants`` place ``data/`` next to ``src/`` so
the notebook runs from a fresh install with no network and no manual copying.
"""
