#!/usr/bin/env python3
"""
The test suite. Run it before spending money.

    python run_tests.py

Nothing here makes an API call or needs a key, and nothing needs agentdojo,
so it runs anywhere the repository does. Standard-library unittest on
purpose: a test suite that needs its own install is a test suite that does
not get run.

Until now this repository had zero tests. Every bug in it was found by
reading a results table and guessing, four of the five diagnoses were wrong,
and two of the bugs had silently invalidated every parallel run that had
ever been done. tests/test_concurrency.py is the one that would have caught
the expensive one.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main() -> int:
    suite = unittest.defaultTestLoader.discover(
        os.path.join(ROOT, "tests"), pattern="test_*.py", top_level_dir=ROOT)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
