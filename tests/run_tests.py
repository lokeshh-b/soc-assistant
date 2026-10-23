"""tests/run_tests.py

Test runner script for the Explainable SOC Assistant.
Runs all unit and integration tests across tests/ without requiring third-party dependencies,
while providing rich formatting and summary metrics.
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tests.test_classifier as tc
import tests.test_integration as ti
import tests.test_parser as tp
import tests.test_streamer as ts


def run_all() -> bool:
    modules = [tp, ts, tc, ti]
    total = 0
    passed = 0
    failed = 0

    print("=" * 70)
    print("RUNNING EXPLAINABLE SOC ASSISTANT TEST SUITE")
    print("=" * 70)
    start_time = time.time()

    for mod in modules:
        mod_name = mod.__name__
        print(f"\n[+] Suite: {mod_name}")
        for name, func in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("test_") and not name.endswith("_placeholder"):
                total += 1
                try:
                    func()
                    print(f"    [PASS] {name}")
                    passed += 1
                except Exception as e:
                    print(f"    [FAIL] {name}: {e}")
                    failed += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed in {elapsed:.3f}s")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
