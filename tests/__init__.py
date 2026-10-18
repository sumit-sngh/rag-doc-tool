"""Test package.

Present so ``tests`` is an importable package: ``tests/test_api.py`` pulls
helpers out of ``tests.conftest``, and with no ``__init__.py`` pytest would put
``tests/`` on ``sys.path`` instead of the project root, breaking both that import
and ``import app`` under a bare ``pytest`` invocation.
"""
