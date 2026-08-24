import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.axisymmetric_reduced_wave_evolution import AxisymmetricVariableReducedWaveIBVP,axisymmetric_bilinear_finite_element_matrices,axisymmetric_bilinear_reaction_matrix,axisymmetric_coupled_lower_order_matrices,axisymmetric_principal_coefficients,axisymmetric_rectangular_lower_order_matrices


class AxisymmetricReducedWaveEvolutionTests(unittest.TestCase):
    def test_rectangular_lower_order_assembly_matches_square_case(self):
        rng=np.random.default_rng(4);z=np.linspace(0,1,5);r=np.linspace(0,1,7)
        weight=1+z[:,None]+.2*r[None,:]
        reaction=rng.normal(size=(5,7,2,2));first=rng.normal(size=(3,5,7,2,2))
        square=axisymmetric_coupled_lower_order_matrices(z,r,weight,reaction,first)
        rectangular=axisymmetric_rectangular_lower_order_matrices(z,r,weight,reaction,first)
        for key in ("reaction","time_first","z_first","r_first"):
            np.testing.assert_allclose(square[key].toarray(),rectangular[key].toarray(),atol=2e-15)

    def test_conformal_principal_coefficients(self):
        psi=np.array(((1.,.8),(.6,.5)));zero=np.zeros_like(psi)
        result=axisymmetric_principal_coefficients(psi,zero,zero,zero)
        np.testing.assert_allclose(result["mass_weight"],psi**3)
        np.testing.assert_allclose(result["z_gradient_weight"],psi**3)
        np.testing.assert_allclose(result["r_gradient_weight"],psi**3)
        np.testing.assert_allclose(result["z_boundary_weight"],psi**4)

    def test_q1_matrices_are_symmetric_and_positive(self):
        z=np.linspace(0,1,7);r=np.linspace(0,1,9);one=np.ones((len(z),len(r)))
        mass,stiffness=axisymmetric_bilinear_finite_element_matrices(z,r,one,one,one)
        self.assertEqual((mass-mass.T).nnz,0)
        self.assertEqual((stiffness-stiffness.T).nnz,0)
        self.assertGreater(np.linalg.eigvalsh(mass.toarray())[0],0.)
        self.assertGreater(np.linalg.eigvalsh(stiffness.toarray())[0],-1e-11)

    def test_full_seventeen_field_manufactured_convergence(self):
        errors=[];omega=1.2;radial_wave=np.pi/2
        for size in (9,17,33,65):
            z=np.linspace(0,1,size);r=np.linspace(0,1,size);shape=(size,size)
            one=np.ones(shape);zero=np.zeros((size,13,13))
            system=AxisymmetricVariableReducedWaveIBVP(
                z,r,one,one,one,zero,zero,np.ones(size),np.ones(size),
            )
            zz,rr=np.meshgrid(z,r,indexing="ij");radial=np.cos(radial_wave*rr)
            spatial=np.zeros((size,size,17));z_second=np.zeros_like(spatial)
            for field in range(4):
                wave=(field+1)*np.pi
                spatial[:,:,field]=np.sin(wave*zz)*radial
                z_second[:,:,field]=-wave**2*spatial[:,:,field]
            for field in range(13):
                wave=(field%3)*np.pi
                spatial[:,:,field+4]=np.cos(wave*zz)*radial
                z_second[:,:,field+4]=-wave**2*spatial[:,:,field+4]
            radial_part=np.empty_like(radial)
            radial_part[:,1:]=-radial_wave**2*radial[:,1:]-2*radial_wave*np.sin(radial_wave*rr[:,1:])/rr[:,1:]
            radial_part[:,0]=-3*radial_wave**2
            source=lambda t,zq,rq:np.cos(omega*t)*(
                -omega**2*spatial-z_second-spatial/radial[:,:,None]*radial_part[:,:,None]
            )
            result=system.integrate(
                spatial,np.zeros_like(spatial),.03,.04,source=source,
            )
            exact=np.cos(omega*.03)*spatial
            errors.append(system.l2_norm(result["position"]-exact)/system.l2_norm(exact))
        rates=np.log(np.asarray(errors[:-1])/np.asarray(errors[1:]))/np.log(2.)
        self.assertGreater(np.min(rates[-2:]),1.8)

    def test_positive_reaction_matrix(self):
        z=np.linspace(0,1,7);r=np.linspace(0,1,9);reaction=1+z[:,None]+r[None,:]
        matrix=axisymmetric_bilinear_reaction_matrix(z,r,reaction)
        self.assertEqual((matrix-matrix.T).nnz,0)
        self.assertGreater(np.linalg.eigvalsh(matrix.toarray())[0],0.)

    def test_massive_scalar_manufactured_convergence(self):
        errors=[];omega=1.1;mass_squared=.7;radial_wave=np.pi/2
        for size in (9,17,33,65):
            z=np.linspace(0,1,size);r=np.linspace(0,1,size);one=np.ones((size,size))
            zero=np.zeros((size,1,1));reaction=(mass_squared*one)[:,:,None]
            system=AxisymmetricVariableReducedWaveIBVP(
                z,r,one,one,one,zero,zero,np.ones(size),np.ones(size),
                dirichlet_fields=0,reaction_weights=reaction,
            )
            zz,rr=np.meshgrid(z,r,indexing="ij");kz=np.pi/2
            spatial=(np.cos(kz*zz)*np.cos(radial_wave*rr))[:,:,None]
            radial_term=np.empty_like(rr)
            radial_term[:,1:]=-radial_wave**2*np.cos(radial_wave*rr[:,1:])-2*radial_wave*np.sin(radial_wave*rr[:,1:])/rr[:,1:]
            radial_term[:,0]=-3*radial_wave**2
            source=lambda t,zq,rq:np.cos(omega*t)*(
                (-omega**2+kz**2+mass_squared)*spatial-radial_term[:,:,None]*np.cos(kz*zz)[:,:,None]
            )
            left_data=lambda t,rq:np.zeros((size,1))
            right_data=lambda t,rq:np.cos(omega*t)*(-kz*np.cos(radial_wave*rq))[:,None]
            result=system.integrate(spatial,np.zeros_like(spatial),.03,.04,source,left_data,right_data)
            exact=np.cos(omega*.03)*spatial
            errors.append(system.l2_norm(result["position"]-exact)/system.l2_norm(exact))
        rates=np.log(np.asarray(errors[:-1])/np.asarray(errors[1:]))/np.log(2.)
        self.assertGreater(np.min(rates[-2:]),1.8)

    def test_coupled_lower_order_manufactured_convergence(self):
        errors=[];omega=1.15;kz=np.pi;kr=np.pi/2
        reaction=np.array(((.4,.1),(-.2,.3)))
        first=np.array((
            ((.05,.01),(0.,-.04)),
            ((.1,.02),(-.03,.05)),
            ((-.06,.01),(.02,.04)),
        ))
        vector=np.array((1.,.7))
        for size in (9,17,33,65):
            z=np.linspace(0,1,size);r=np.linspace(0,1,size);one=np.ones((size,size))
            zero_wall=np.zeros((size,2,2))
            reaction_field=np.broadcast_to(reaction,(size,size,2,2)).copy()
            first_field=np.broadcast_to(first[:,None,None],(3,size,size,2,2)).copy()
            system=AxisymmetricVariableReducedWaveIBVP(
                z,r,one,one,one,zero_wall,zero_wall,np.ones(size),np.ones(size),
                dirichlet_fields=0,coupled_reaction_matrices=reaction_field,
                evolution_first_matrices=first_field,
            )
            zz,rr=np.meshgrid(z,r,indexing="ij");base=np.cos(kz*zz)*np.cos(kr*rr)
            uz=-kz*np.sin(kz*zz)*np.cos(kr*rr);ur=-kr*np.cos(kz*zz)*np.sin(kr*rr)
            lap=(-kz**2-kr**2)*base
            lap[:,1:]+=2*ur[:,1:]/rr[:,1:]
            lap[:,0]=-kz**2*base[:,0]-3*kr**2*np.cos(kz*z)
            spatial=base[:,:,None]*vector;z_first=uz[:,:,None]*vector
            r_first=ur[:,:,None]*vector;laplacian=lap[:,:,None]*vector
            def source(time,zq,rq):
                cosine=np.cos(omega*time);sine=np.sin(omega*time)
                value=cosine*spatial;velocity=-omega*sine*spatial
                desired=-omega**2*value
                lower=cosine*(np.einsum("ab,ijb->ija",first[1],z_first)+np.einsum("ab,ijb->ija",first[2],r_first))
                lower+=np.einsum("ab,ijb->ija",first[0],velocity)
                return desired-cosine*laplacian-lower+np.einsum("ab,ijb->ija",reaction,value)
            result=system.integrate(spatial,np.zeros_like(spatial),.025,.04,source=source)
            exact=np.cos(omega*.025)*spatial
            errors.append(system.l2_norm(result["position"]-exact)/system.l2_norm(exact))
        rates=np.log(np.asarray(errors[:-1])/np.asarray(errors[1:]))/np.log(2.)
        self.assertGreater(np.min(rates[-2:]),1.8)

    def test_scaled_radial_first_coefficients_match_regular_constant_field(self):
        z=np.linspace(0,1,5);r=np.linspace(0,1,7);one=np.ones((5,7))
        reaction=np.zeros((5,7,1,1));first=np.zeros((3,5,7,1,1));first[2,:,:,0,0]=.4
        direct=axisymmetric_coupled_lower_order_matrices(z,r,one,reaction,first)
        scaled=first.copy();scaled[2,:,:,0,0]*=r[None,:]
        regular=axisymmetric_coupled_lower_order_matrices(
            z,r,one,reaction,scaled,radial_first_is_scaled=True,
        )
        np.testing.assert_allclose(direct["r_first"].toarray(),regular["r_first"].toarray(),atol=2e-15)

    def test_variable_coupled_lower_order_scaled_radial_convergence(self):
        errors=[];vector=np.array((1.,-.4));omega=1.2;kz=np.pi;kr=np.pi/2
        for size in (9,17,33,65):
            z=np.linspace(0,1,size);r=np.linspace(0,1,size);zz,rr=np.meshgrid(z,r,indexing="ij")
            one=np.ones((size,size));zero_wall=np.zeros((size,2,2))
            reaction=np.zeros((size,size,2,2));first=np.zeros((3,size,size,2,2))
            reaction[:,:,0,0]=.3+.1*zz;reaction[:,:,0,1]=.05*rr**2
            reaction[:,:,1,0]=-.04*zz;reaction[:,:,1,1]=.2+.03*rr**2
            first[0,:,:,0,0]=.02*zz;first[0,:,:,1,1]=-.03*zz
            first[1,:,:,0,1]=.04*(1+rr**2);first[1,:,:,1,0]=-.02*(1+zz)
            # Store r B^r, including its regular zero at the axis.
            first[2,:,:,0,0]=rr*(.06+.01*zz);first[2,:,:,1,1]=rr*(-.04+.02*zz)
            system=AxisymmetricVariableReducedWaveIBVP(
                z,r,one,one,one,zero_wall,zero_wall,np.ones(size),np.ones(size),
                dirichlet_fields=0,coupled_reaction_matrices=reaction,
                evolution_first_matrices=first,radial_first_is_scaled=True,
            )
            base=np.cos(kz*zz)*np.cos(kr*rr);uz=-kz*np.sin(kz*zz)*np.cos(kr*rr)
            ur=-kr*np.cos(kz*zz)*np.sin(kr*rr);lap=(-kz**2-kr**2)*base
            lap[:,1:]+=2*ur[:,1:]/rr[:,1:];lap[:,0]=-kz**2*base[:,0]-3*kr**2*np.cos(kz*z)
            values=base[:,:,None]*vector;z_first=uz[:,:,None]*vector
            r_first=ur[:,:,None]*vector;laplacian=lap[:,:,None]*vector
            radial_lower=np.zeros_like(values)
            radial_lower[:,1:]=np.einsum("ijab,ijb->ija",first[2,:,1:]/rr[:,1:,None,None],r_first[:,1:])
            radial_lower[:,0]=0.
            spatial_lower=np.einsum("ijab,ijb->ija",first[1],z_first)+radial_lower
            reaction_term=np.einsum("ijab,ijb->ija",reaction,values)
            def source(time,zq,rq):
                cosine=np.cos(omega*time);sine=np.sin(omega*time)
                velocity=-omega*sine*values
                time_lower=np.einsum("ijab,ijb->ija",first[0],velocity)
                return -omega**2*cosine*values-cosine*laplacian-cosine*spatial_lower-time_lower+cosine*reaction_term
            result=system.integrate(values,np.zeros_like(values),.02,.035,source=source)
            exact=np.cos(omega*.02)*values
            errors.append(system.l2_norm(result["position"]-exact)/system.l2_norm(exact))
        rates=np.log(np.asarray(errors[:-1])/np.asarray(errors[1:]))/np.log(2.)
        self.assertGreater(rates[-1],1.8)

    def test_live_outer_trace_keeps_radial_boundary_in_mass_solve(self):
        z=np.linspace(0,1,7);r=np.linspace(0,1,9);one=np.ones((len(z),len(r)))
        wall=np.zeros((len(r),1,1))
        hard=AxisymmetricVariableReducedWaveIBVP(
            z,r,one,one,one,wall,wall,np.ones(len(r)),np.ones(len(r)),
            dirichlet_fields=0,outer_dirichlet=True,
        )
        live=AxisymmetricVariableReducedWaveIBVP(
            z,r,one,one,one,wall,wall,np.ones(len(r)),np.ones(len(r)),
            dirichlet_fields=0,outer_dirichlet=False,
        )
        outer=np.arange(live.nodes).reshape(live.nz,live.nr)[:,-1]
        self.assertTrue(np.all(np.isin(outer,hard.robin_fixed)))
        self.assertTrue(np.all(np.isin(outer,live.robin_free)))
        self.assertEqual(len(live.robin_fixed),0)
        constant=np.ones((live.nz,live.nr,1))
        np.testing.assert_allclose(
            live.acceleration(0.,constant),0.,atol=3e-13,
        )
        result=live.integrate(
            np.zeros_like(constant),np.zeros_like(constant),.01,
        )
        np.testing.assert_allclose(result["position"],0.)
        np.testing.assert_allclose(result["velocity"],0.)

    def test_velocity_diffusion_has_nonpositive_semidiscrete_energy_power(self):
        z=np.linspace(0,1,7);r=np.linspace(0,1,9);one=np.ones((len(z),len(r)))
        wall=np.zeros((len(r),1,1));coefficient=.17
        system=AxisymmetricVariableReducedWaveIBVP(
            z,r,one,one,one,wall,wall,np.ones(len(r)),np.ones(len(r)),
            dirichlet_fields=0,outer_dirichlet=False,
            velocity_diffusion=coefficient,
        )
        zz,rr=np.meshgrid(z,r,indexing="ij")
        position=np.zeros((len(z),len(r),1))
        velocity=(np.cos(np.pi*zz)*np.cos(np.pi*rr/2))[:,:,None]
        acceleration=system.acceleration(0.,position,velocity=velocity)
        flat_velocity=velocity.reshape(system.nodes,1)
        flat_acceleration=acceleration.reshape(system.nodes,1)
        measured=float(np.sum(flat_velocity*(system.mass@flat_acceleration)))
        expected=-coefficient*float(np.sum(flat_velocity*(system.stiffness@flat_velocity)))
        self.assertLess(measured,0.)
        self.assertAlmostEqual(measured,expected,places=12)

    def test_outer_radial_flux_load_integrates_surface_and_excludes_corners(self):
        z=np.linspace(0,1,5);r=np.linspace(0,2,7);one=np.ones((len(z),len(r)))
        wall=np.zeros((len(r),1,1));system=AxisymmetricVariableReducedWaveIBVP(
            z,r,one,one,one,wall,wall,np.ones(len(r)),np.ones(len(r)),
            dirichlet_fields=0,outer_dirichlet=False,
        )
        active=np.ones(len(z),dtype=bool);active[[0,-1]]=False
        load=system.outer_radial_flux_load(np.ones((len(z),1)),active)
        outer=np.arange(system.nodes).reshape(system.nz,system.nr)[:,-1]
        interior=outer[1:-1]
        self.assertAlmostEqual(float(np.sum(load)),3.)
        np.testing.assert_allclose(load[outer[[0,-1]]],0.)
        self.assertTrue(np.all(load[interior]>0))


if __name__=="__main__":unittest.main()
