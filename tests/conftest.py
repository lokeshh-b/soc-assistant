"""tests/conftest.py

Pytest configuration and environment setup.
Ensures the project workspace root is on sys.path so modules in src/ are importable.
"""

import sys
from pathlib import Path

# Ensure project root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
