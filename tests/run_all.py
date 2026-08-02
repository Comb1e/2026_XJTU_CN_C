"""Unified test runner for all Problem 1 tests.

Usage:
    python tests/run_all.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


def main() -> None:
    test_dir = Path(__file__).resolve().parent
    args = [
        str(test_dir),
        "-v",
        "--tb=short",
        "-p", "no:warnings",
    ]
    exit_code = pytest.main(args)
    if exit_code == pytest.ExitCode.OK:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ Tests failed with exit code {exit_code}")
    sys.exit(exit_code.value)


if __name__ == "__main__":
    main()
