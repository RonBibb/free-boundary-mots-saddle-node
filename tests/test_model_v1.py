import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ModelV1Tests(unittest.TestCase):
    def test_model_uses_invariant_energy(self):
        text = (ROOT / "MODEL_V1.md").read_text()
        self.assertIn("mathcal E", text)
        self.assertIn("kappa_5^2E_\\chi", text)

    def test_experiment_matrix_has_controls_and_primary_model(self):
        with (ROOT / "EXPERIMENT_MATRIX_V1.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(any(row["model"] == "C1" for row in rows))
        self.assertTrue(any(row["model"] == "V1" for row in rows))

    def test_claim_ledger_records_model_selection(self):
        claims = json.loads((ROOT / "CLAIM_LEDGER.json").read_text())
        self.assertIn("primary_model_selected", claims[0]["status"])

    def test_orbifold_action_and_boundary_conventions_are_consistent(self):
        model=(ROOT/"MODEL_V1.md").read_text();equations=(ROOT/"EQUATIONS_V1.md").read_text()
        self.assertIn("doubled covering space",model)
        self.assertIn("+\\frac{\\kappa_5^2}{2}",equations)
        self.assertIn("-\\frac12U_i'",equations)


if __name__ == "__main__":
    unittest.main()
