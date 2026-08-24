import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.vaidya_horizon_control import outgoing_expansion,outgoing_null_rhs,thin_shell_apparent_horizon,thin_shell_event_horizon,thin_shell_mass,trace_smooth_event_horizon


class VaidyaHorizonControlTests(unittest.TestCase):
    def test_thin_shell_event_horizon_is_exact_outgoing_null_generator(self):
        mass=1.3;shell=.4;v=np.linspace(shell-4*mass+.1,shell-.1,101)
        radius=thin_shell_event_horizon(v,mass,shell)
        derivative=np.gradient(radius,v,edge_order=2)
        rhs=outgoing_null_rhs(v,radius,thin_shell_mass(v,mass,shell))
        np.testing.assert_allclose(derivative,rhs,atol=2e-13)

    def test_event_horizon_precedes_shell_and_is_not_marginal_there(self):
        mass=2.;shell=1.;sample=np.array((shell-4*mass,shell-2*mass,shell-.1,shell+.1))
        event=thin_shell_event_horizon(sample,mass,shell);apparent=thin_shell_apparent_horizon(sample,mass,shell)
        self.assertAlmostEqual(event[0],0.)
        self.assertTrue(np.all(np.isnan(apparent[:3])))
        expansion=outgoing_expansion(event[1:3],np.zeros(2))
        self.assertTrue(np.all(expansion>0))
        self.assertAlmostEqual(event[-1],apparent[-1])

    def test_smooth_event_horizon_is_outside_apparent_horizon_and_starts_early(self):
        solved=trace_smooth_event_horizon(final_mass=1.,start=0.,duration=1.)
        self.assertTrue(solved["converged"])
        self.assertLess(solved["birth_time"],solved["collapse_start"])
        self.assertGreater(solved["minimum_event_minus_apparent"],-2e-9)
        self.assertAlmostEqual(solved["event_radius"][-1],2.,places=10)
        before=solved["v"]<0
        line=.5*(solved["v"][before]-solved["birth_time"])
        np.testing.assert_allclose(solved["event_radius"][before],line,atol=2e-7)


if __name__=="__main__":unittest.main()
