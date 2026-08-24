import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.israel_wave_matrix import analytic_robin_symmetrizer,coupled_robin_matrix
from bhps.reduced_wave_evolution import FrozenReducedWaveIBVP,VariableCoefficientReducedWaveIBVP,anisotropic_wave_principal_coefficients,linear_finite_element_matrices,variable_linear_finite_element_matrices


class ReducedWaveEvolutionTests(unittest.TestCase):
    def test_finite_element_matrices_are_symmetric_positive(self):
        mass,stiffness=linear_finite_element_matrices(np.linspace(0,1,17))
        self.assertLess(np.linalg.norm(mass-mass.T),1e-14)
        self.assertLess(np.linalg.norm(stiffness-stiffness.T),1e-14)
        self.assertGreater(np.linalg.eigvalsh(mass)[0],0.)
        self.assertGreater(np.linalg.eigvalsh(stiffness)[0],-1e-12)

    def test_zero_full_seventeen_field_state_is_preserved(self):
        robin=coupled_robin_matrix(1.,0.,0.,20.)["matrix"]
        system=FrozenReducedWaveIBVP(np.linspace(0,1,25),robin,robin)
        zero=np.zeros((25,17))
        result=system.integrate(zero,zero,.05)
        self.assertEqual(result["position"].shape,(25,17))
        self.assertEqual(np.max(np.abs(result["position"])),0.)
        self.assertEqual(np.max(np.abs(result["velocity"])),0.)

    def test_mirrored_physical_robin_energy_is_conserved(self):
        audit=analytic_robin_symmetrizer(1.,.001,.003,20.)
        robin=audit["matrix"]
        points=np.linspace(0,1,49)
        system=FrozenReducedWaveIBVP(points,robin,robin)
        position=np.zeros((len(points),17));velocity=np.zeros_like(position)
        position[:,4+12]=np.cos(np.pi*points)
        energy=lambda t,q,p:system.symmetrized_energy(q,p,audit["symmetrizer"])["total"]
        result=system.integrate(position,velocity,.1,courant=.15,diagnostic=energy)
        records=np.asarray(result["diagnostics"])
        self.assertLess(np.max(np.abs(records-records[0]))/abs(records[0]),1e-7)

    def test_dirichlet_constraint_mode_converges(self):
        errors=[]
        for size in (25,49,97):
            points=np.linspace(0,1,size);robin=np.zeros((4,4))
            system=FrozenReducedWaveIBVP(points,robin,robin,dirichlet_fields=1)
            position=np.zeros((size,5));velocity=np.zeros_like(position)
            position[:,0]=np.sin(np.pi*points)
            result=system.integrate(position,velocity,.2,courant=.2)
            exact=np.zeros_like(position)
            exact[:,0]=np.cos(.2*np.pi)*np.sin(np.pi*points)
            errors.append(system.l2_norm(result["position"]-exact))
        rates=np.log(np.asarray(errors[:-1])/np.asarray(errors[1:]))/np.log(2.)
        self.assertGreater(np.min(rates),1.8)

    def test_anisotropic_principal_coefficients_reduce_correctly(self):
        psi=np.array((1.,.5));zero=np.zeros(2)
        result=anisotropic_wave_principal_coefficients(psi,zero,zero,zero)
        np.testing.assert_allclose(result["mass_weight"],psi**3)
        np.testing.assert_allclose(result["gradient_weight"],psi**3)
        np.testing.assert_allclose(result["boundary_weight"],psi**4)
        np.testing.assert_allclose(result["coordinate_speed"],1.)

    def test_variable_matrices_reduce_to_constant_matrices(self):
        points=np.linspace(0,1,17);constant=np.ones(17)
        expected=linear_finite_element_matrices(points)
        actual=variable_linear_finite_element_matrices(points,constant,constant)
        np.testing.assert_allclose(actual[0],expected[0])
        np.testing.assert_allclose(actual[1],expected[1])

    def test_variable_coefficient_manufactured_solution_converges(self):
        errors=[];omega=1.3
        for size in (25,49,97):
            points=np.linspace(0,1,size);w=1+points;p=1+.2*points
            zero=np.zeros((1,1))
            system=VariableCoefficientReducedWaveIBVP(
                points,w,p,zero,zero,1.,1.2,dirichlet_fields=0,
            )
            spatial=(1+points+points**2)[:,None]
            source=lambda t,x:np.cos(omega*t)*(
                -omega**2*(1+x+x**2)-(2.2+.8*x)/(1+x)
            )[:,None]
            left=lambda t:np.array((-np.cos(omega*t),))
            right=lambda t:np.array((3*np.cos(omega*t),))
            result=system.integrate(
                spatial,np.zeros_like(spatial),.15,.18,source,left,right,
            )
            exact=np.cos(omega*.15)*spatial
            errors.append(system.l2_norm(result["position"]-exact))
        rates=np.log(np.asarray(errors[:-1])/np.asarray(errors[1:]))/np.log(2.)
        self.assertGreater(np.min(rates),1.8)


if __name__=="__main__":unittest.main()
