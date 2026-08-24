import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.lapse_corner_repair import lapse_log_basis


class LapseCornerRepairTests(unittest.TestCase):
    def test_basis_preserves_compact_neumann_condition(self):
        z=np.linspace(1,np.e,65);r=np.linspace(0,8,81)
        basis=lapse_log_basis(z,r,4,6)["basis"]
        dz=z[1]-z[0]
        lower=(-3*basis[:,0]+4*basis[:,1]-basis[:,2])/(2*dz)
        upper=(3*basis[:,-1]-4*basis[:,-2]+basis[:,-3])/(2*dz)
        # Second-order numerical differentiation of an exactly Neumann cosine.
        self.assertLess(np.max(np.abs(lower)),.02)
        self.assertLess(np.max(np.abs(upper)),.02)

    def test_basis_vanishes_at_outer_radial_boundary(self):
        z=np.linspace(1,np.e,17);r=np.linspace(0,8,21)
        basis=lapse_log_basis(z,r,3,5)["basis"]
        self.assertLess(np.max(np.abs(basis[:,:,-1])),1e-12)


if __name__=="__main__":unittest.main()
