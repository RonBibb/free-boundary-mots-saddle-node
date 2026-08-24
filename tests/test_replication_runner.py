import ast,unittest
from pathlib import Path

class ReplicationRunnerTests(unittest.TestCase):
    def test_runner_has_three_grids_and_two_domains(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/"run_v1_replication.py").read_text())
        text=ast.unparse(tree)
        self.assertIn("G1_R8",text);self.assertIn("G2_R8",text);self.assertIn("G3_R8",text)
        self.assertIn("G2_R10",text);self.assertIn("G3_R10",text)

if __name__=="__main__":unittest.main()
