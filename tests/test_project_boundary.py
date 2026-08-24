import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectBoundaryTests(unittest.TestCase):
    def test_claims_do_not_assume_common_horizon(self):
        claims = json.loads((ROOT / "CLAIM_LEDGER.json").read_text())
        spanning = next(item for item in claims if item["id"] == "BHPS-C04")
        self.assertNotEqual(spanning["status"], "established")

    def test_dynamic_claim_remains_untested(self):
        claims = json.loads((ROOT / "CLAIM_LEDGER.json").read_text())
        dynamics = next(item for item in claims if item["id"] == "BHPS-C05")
        self.assertEqual(dynamics["status"], "open")

    def test_model_definition_precedes_mechanism_claims(self):
        claims = json.loads((ROOT / "CLAIM_LEDGER.json").read_text())
        self.assertEqual(claims[0]["id"], "BHPS-C00")
        self.assertIn("wellposedness_audit_open", claims[0]["status"])


if __name__ == "__main__":
    unittest.main()
