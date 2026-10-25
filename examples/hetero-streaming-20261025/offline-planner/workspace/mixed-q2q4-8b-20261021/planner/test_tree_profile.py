import json
import tempfile
import unittest
from pathlib import Path

from tree_profile import _cpu_priors, _upper_envelope, export_profile, _training_residual


class TestTreeProfile(unittest.TestCase):
    def test_cpu_priors_are_strict(self):
        with self.assertRaises(ValueError):
            _cpu_priors({"cpu_batch_priors_ms": [1.0] * 15})
        with self.assertRaises(ValueError):
            _cpu_priors({"cpu_batch_priors_ms": [0.0] * 16})

    def test_envelope_never_drops(self):
        self.assertEqual(_upper_envelope([2.0, 1.0, 3.0, 2.0]), [2.0, 2.0, 3.0, 3.0])

    def test_residual_uses_each_training_mixture_not_target(self):
        root=Path(__file__).resolve().parent
        table=json.loads((root/'output/cost-table.json').read_text())['layers']
        a,_,_=_training_residual(table,[0]*24,1.)
        b,_,_=_training_residual(table,[32]*24,2.)
        self.assertEqual(a,b)

    def test_export_real_frozen_inputs(self):
        root = Path(__file__).resolve().parent
        table = root / "output" / "cost-table.json"
        plan = json.loads((root / "output" / "plan.json").read_text())
        cpu = root.parent / "tree-initial-146" / "initial-costs.json"
        with tempfile.TemporaryDirectory() as temp:
            manifest = export_profile(table, plan, cpu, temp)
            lines = (Path(temp) / "initial-profile.tsv").read_text().splitlines()
            self.assertEqual(sum(line.startswith("V ") for line in lines), 64)
            self.assertEqual(sum(line.startswith("C ") for line in lines), 16)
            values = [float(line.split()[2]) for line in lines if line.startswith("V ")]
            self.assertEqual(values, sorted(values))
            self.assertEqual(manifest["cpu"]["extrapolated_batches"], list(range(9, 17)))


if __name__ == "__main__":
    unittest.main()
