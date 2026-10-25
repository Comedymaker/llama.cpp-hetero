import sys
import unittest
from pathlib import Path

source = Path(__file__).resolve().parent / 'workspace/mixed-q2q4-8b-20261021/planner'
sys.path.insert(0, str(source))
suite = unittest.defaultTestLoader.discover(str(source), pattern='test_*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
