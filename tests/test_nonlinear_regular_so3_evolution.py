import unittest
from unittest import mock
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.nonlinear_regular_so3_evolution import (
    GaugeTaylorSource,
    NativeRegularSO3RHS,
    StageRegularGaugeSource,
    apply_compact_wall_acceleration,
    apply_outer_sommerfeld_acceleration,
    apply_outer_source_sommerfeld,
    compact_wall_normal_gauge_acceleration_residuals,
    fill_regular_axis,
    outer_sommerfeld_position_residuals,
    reconcile_wall_owner_axis_null_channels,
    regular_so3_outward_radial_speed,
    solve_compact_wall_coupled_phi_normal_acceleration,
    solve_compact_wall_normal_gauge_acceleration,
)
from bhps.junction_preservation_diagnostic import manufactured_state
from bhps.junction_second_preservation_diagnostic import (
    wall_junction_second_tangent,
)
from bhps.staged_boundary_preservation import evaluate_boundary_stage_sequence


class NonlinearRegularSO3EvolutionTests(unittest.TestCase):
    def test_even_axis_fit(self):
        r = np.linspace(0, 1, 13)
        field = np.zeros((2, len(r), 1))
        field[:, 1:, 0] = 2 + 3 * r[None, 1:] ** 2 - r[None, 1:] ** 4
        fitted = fill_regular_axis(field, r)
        self.assertTrue(np.allclose(fitted[:, 0, 0], 2, atol=1e-12))

    def test_wall_owner_axis_quotients_use_native_numerator_images(self):
        r = np.linspace(0.0, 1.0, 17)
        value = np.zeros((4, len(r), 9))
        value[:, 0, (1, 4, 5)] = 99.0
        value[:, 1:, 1] = 2.0 + 0.3 * r[None, 1:] ** 2
        value[:, 1:, 4] = -1.0 + 0.2 * r[None, 1:] ** 2
        value[:, 1:, 5] = 0.25 - 0.1 * r[None, 1:] ** 2
        found = reconcile_wall_owner_axis_null_channels(value, r)
        np.testing.assert_allclose(found[:, 0, 1], 2.0, atol=2e-13)
        np.testing.assert_allclose(found[:, 0, 4], -1.0, atol=2e-13)
        np.testing.assert_allclose(found[:, 0, 5], 0.25, atol=2e-13)
        np.testing.assert_array_equal(found[:, 1:], value[:, 1:])

    def test_gauge_taylor_source(self):
        z = np.linspace(1, 2, 9)
        r = np.linspace(0, 1, 13)
        value = np.zeros((len(z), len(r), 5))
        first = np.zeros((len(z), len(r), 5, 5))
        zz, rr = np.meshgrid(z, r, indexing="ij")
        first[:, :, 0, 0] = 1 + zz
        first[:, :, 0, 1] = 2 - rr**2
        first[:, :, 0, 2] = rr * (3 + zz)
        source = GaugeTaylorSource(value, first, z, r)
        h, hf = source.at(3, 4, 0.2)
        self.assertTrue(np.allclose(h, 0.2 * first[3, 4, 0]))
        self.assertTrue(np.allclose(hf[0], first[3, 4, 0]))

    def test_live_stage_source_maps_regular_time_and_spatial_jets(self):
        z=np.linspace(1,2,9);r=np.linspace(0,1,13)
        zz,rr=np.meshgrid(z,r,indexing="ij")
        source=np.empty((len(z),len(r),3))
        source[:,:,0]=zz+rr**2
        source[:,:,1]=2*zz-rr**2
        source[:,:,2]=.3+.1*zz+.2*rr**2
        source_time=np.full_like(source,.07)
        stage=StageRegularGaugeSource(source,source_time,z,r)
        value,first=stage.at(4,6)
        np.testing.assert_allclose(value,(source[4,6,0],source[4,6,1],r[6]*source[4,6,2],0.,0.))
        np.testing.assert_allclose(first[0,0:2],(.07,.07),atol=1e-13)
        self.assertAlmostEqual(first[0,2],r[6]*.07)
        self.assertAlmostEqual(first[3,3],source[4,6,2])

    def test_time_symmetric_wall_solve_closes_manufactured_rows(self):
        z = np.linspace(1, 2, 9)
        r = np.linspace(0, 1, 13)
        q = np.zeros((len(z), len(r), 9))
        v = np.zeros_like(q)
        a = np.zeros_like(q)
        q[:, :, 2] = -1
        q[:, :, 3] = 1
        q[:, :, 6] = 1
        q[:, :, 7] = np.linspace(-0.2, 0.2, len(z))[:, None]
        background = {
            "wall_stiffness": 2.0, "v0": -0.2, "v1": 0.2,
            "beta_a": 0.0, "beta_b": 0.0,
            "wall_potential_a": 0.0, "wall_potential_b": 0.0,
        }
        normal = np.vstack((np.ones(len(r)), -np.ones(len(r))))
        solved, _ = apply_compact_wall_acceleration(
            q, v, a, z, r, background, normal,
        )
        self.assertTrue(np.all(np.isfinite(solved)))
        self.assertTrue(np.allclose(solved[[0, -1], :, 0], 0))
        self.assertTrue(np.allclose(solved[[0, -1], :, 1], 0))

    def test_normal_gauge_acceleration_solve_closes_manufactured_row(self):
        z=np.linspace(1,2,9);r=np.linspace(0,1,13)
        q=np.zeros((len(z),len(r),9));v=np.zeros_like(q);a=np.zeros_like(q)
        q[:,:,2]=-1.;q[:,:,3]=1.;q[:,:,6]=1.
        source=np.zeros((len(z),len(r),3));source_t=np.zeros_like(source)
        source_tt=np.zeros_like(source)
        background={
            "wall_stiffness":2.,"v0":0.,"v1":0.,
            "beta_a":0.,"beta_b":0.,
            "wall_potential_a":0.,"wall_potential_b":0.,
        }
        zz,rr=np.meshgrid(z,r,indexing="ij")
        a[:,:,6]=(zz-1.5)**2*(1+.2*rr**2)
        solved,_=solve_compact_wall_normal_gauge_acceleration(
            q,v,a,source,source_t,source_tt,z,r,background,
        )
        residual=compact_wall_normal_gauge_acceleration_residuals(
            q,v,solved,source,source_t,source_tt,z,r,background,radial_buffer=0,
        )
        self.assertLess(residual["maximum"],1e-11)

    def test_flat_outer_speed_is_one(self):
        r=np.linspace(0,2,17);q=np.zeros((7,len(r),9))
        q[:,:,2]=-1.;q[:,:,3]=1.;q[:,:,6]=1.
        np.testing.assert_allclose(regular_so3_outward_radial_speed(q,r),1.,atol=1e-13)

    def test_outer_sommerfeld_closes_manufactured_outgoing_row(self):
        z=np.linspace(1,2,9);r=np.linspace(0,2,17)
        q0=np.zeros((len(z),len(r),9));q0[:,:,2]=-1.;q0[:,:,3]=1.;q0[:,:,6]=1.
        a0=np.zeros_like(q0);time=.13
        zz,rr=np.meshgrid(z,r,indexing="ij")
        profile=(rr-time)**3*(1+.1*zz)
        q=q0.copy();v=np.zeros_like(q);a=np.zeros_like(q)
        q[:,:,8]=profile
        v[:,:,8]=-3*(rr-time)**2*(1+.1*zz)
        a[:,:,8]=6*(rr-time)*(1+.1*zz)
        solved,diagnostic=apply_outer_sommerfeld_acceleration(
            q,v,a,q0,a0,time,r,
        )
        expected=6*(r[-1]-time)*(1+.1*z[1:-1])
        np.testing.assert_allclose(solved[1:-1,-1,8],expected,rtol=1e-10,atol=1e-10)
        self.assertLess(diagnostic["maximum_normalized_acceleration_residual"],1e-12)
        residual=outer_sommerfeld_position_residuals(q,v,q0,a0,time,r)
        self.assertLess(residual["maximum_normalized"],1e-11)

    def test_outer_source_sommerfeld_preserves_reference_jet(self):
        z=np.linspace(1,2,9);r=np.linspace(0,2,17)
        q=np.zeros((len(z),len(r),9));q[:,:,2]=-1.;q[:,:,3]=1.;q[:,:,6]=1.
        h0=np.zeros((len(z),len(r),3));ht0=np.full_like(h0,.2);htt0=np.full_like(h0,-.03)
        time=.1;h=h0+time*ht0+.5*time*time*htt0
        trial=np.full_like(h,7.)
        solved,diagnostic=apply_outer_source_sommerfeld(
            h,trial,h0,ht0,htt0,q,time,r,
        )
        np.testing.assert_allclose(solved[1:-1,-1],ht0[1:-1,-1]+time*htt0[1:-1,-1])
        self.assertLess(diagnostic["maximum_normalized"],1e-12)

    def test_staged_wall_refactor_retains_legacy_default_exactly(self):
        z=np.linspace(1,2,17);r=np.linspace(0,1,25)
        zz,rr=np.meshgrid(z,r,indexing="ij")
        q=np.zeros((len(z),len(r),9));v=np.zeros_like(q);a=np.zeros_like(q)
        q[:,:,2]=-1.;q[:,:,3]=1.;q[:,:,6]=1.
        q[:,:,7]=.02*zz
        for field in range(9):
            a[:,:,field]=(.01+.002*field)*(1+.1*zz)*np.exp(.2*rr**2)
        background={
            "wall_stiffness":2.,"v0":.02,"v1":.04,
            "beta_a":.1,"beta_b":.12,
            "wall_potential_a":0.,"wall_potential_b":0.,
        }
        normal=np.stack((a[0,:,6],a[-1,:,6]))
        legacy,_=apply_compact_wall_acceleration(
            q,v,a,z,r,background,normal,
        )
        prefill,_=apply_compact_wall_acceleration(
            q,v,a,z,r,background,normal,fill_axis_after=False,
        )
        composed=fill_regular_axis(prefill,r)
        composed[0,0,6]=normal[0,0];composed[-1,0,6]=normal[1,0]
        np.testing.assert_array_equal(legacy,composed)

    def test_native_trace_is_observational_and_bitwise_identical(self):
        z=np.linspace(1,2,9);r=np.linspace(0,1,13)
        q=np.zeros((len(z),len(r),9));v=np.zeros_like(q)
        q[:,:,2]=-1.;q[:,:,3]=1.;q[:,:,6]=1.
        source_value=np.zeros((len(z),len(r),5))
        source_first=np.zeros((len(z),len(r),5,5))
        source=GaugeTaylorSource(source_value,source_first,z,r)
        background={
            "wall_stiffness":0.,"v0":0.,"v1":0.,
            "beta_a":0.,"beta_b":0.,
            "wall_potential_a":0.,"wall_potential_b":0.,
        }
        rhs=NativeRegularSO3RHS(
            z,r,source,0.,background,np.zeros((2,len(r))),
        )
        plain,plain_diagnostic=rhs.acceleration(0.,q,v)
        traced,traced_diagnostic=rhs.acceleration(
            0.,q,v,capture_boundary_stages=True,
        )
        np.testing.assert_array_equal(plain,traced)
        self.assertIsNone(plain_diagnostic["boundary_stages"])
        self.assertEqual(
            traced_diagnostic["boundary_stage_names"],
            [
                "bulk_positive_radius","initial_axis_fill",
                "final_compact_wall_endpoint_solve",
                "final_compact_post_wall_axis_fill","pre_outer",
                "post_axis_operator_repair",
            ],
        )
        traced_diagnostic["boundary_stages"][0]["acceleration"][:]=123.
        np.testing.assert_array_equal(plain,traced)

        rhs.set_outer_sommerfeld_reference(q,plain)
        plain_outer,_=rhs.acceleration(0.,q,v)
        traced_outer,outer_diagnostic=rhs.acceleration(
            0.,q,v,capture_boundary_stages=True,
        )
        np.testing.assert_array_equal(plain_outer,traced_outer)
        self.assertEqual(outer_diagnostic["boundary_stage_names"][-3:],[
            "pre_outer","post_outer","post_axis_operator_repair",
        ])

    def test_native_wall_owner_last_closes_outer_corner(self):
        z=np.linspace(1,2,9);r=np.linspace(0,1,13)
        q=np.zeros((len(z),len(r),9));v=np.zeros_like(q)
        q[:,:,2]=-1.;q[:,:,3]=1.;q[:,:,6]=1.
        source=GaugeTaylorSource(
            np.zeros((len(z),len(r),5)),
            np.zeros((len(z),len(r),5,5)),z,r,
        )
        background={
            "wall_stiffness":0.,"v0":0.,"v1":0.,
            "beta_a":0.,"beta_b":0.,
            "wall_potential_a":0.,"wall_potential_b":0.,
        }
        common=dict(
            z=z,r=r,gauge_source=source,mass_squared=0.,
            background=background,
            normal_wall_acceleration=np.zeros((2,len(r))),
            live_outer_sommerfeld=True,
        )
        legacy=NativeRegularSO3RHS(
            **common,boundary_closure_mode="legacy_wall_axis_outer",
        )
        owner=NativeRegularSO3RHS(
            **common,boundary_closure_mode="wall_owner_last_experimental",
        )
        for rhs in (legacy,owner):
            rhs.outer_reference_position=q.copy()
            rhs.outer_reference_acceleration=np.zeros_like(q)
        legacy_acceleration,_=legacy.acceleration(0.,q,v)
        owner_acceleration,diagnostic=owner.acceleration(
            0.,q,v,capture_boundary_stages=True,
        )
        legacy_outer=0.;owner_outer=0.
        for wall in ("lower","upper"):
            legacy_record=wall_junction_second_tangent(
                q,v,legacy_acceleration,z,r,background,wall,
            )
            owner_record=wall_junction_second_tangent(
                q,v,owner_acceleration,z,r,background,wall,
            )
            legacy_outer=max(
                legacy_outer,np.linalg.norm(legacy_record["DX2J_tensor"][-1]),
            )
            owner_outer=max(
                owner_outer,np.linalg.norm(owner_record["DX2J_tensor"][-1]),
            )
        self.assertGreater(legacy_outer,100.)
        self.assertLess(owner_outer,1e-10)
        self.assertLess(
            diagnostic["outer_sommerfeld"][
                "maximum_normalized_acceleration_residual"
            ],1e-12,
        )
        stages=diagnostic["boundary_stages"]
        outer_stage=next(
            stage["acceleration"] for stage in stages
            if stage["name"]=="outer_open_face_before_wall"
        )
        # The outer owner remains exact at every positive-radius point.  The
        # open-z axis is intentionally replaced only after all boundary owners
        # by the regular analytic pointwise/native-quotient operator.
        np.testing.assert_array_equal(
            owner_acceleration[1:-1,1:],outer_stage[1:-1,1:],
        )

    def test_native_wall_owner_last_integrates_live_normal_coupled_block(self):
        z=np.linspace(1.,2.,9);r=np.linspace(0.,1.,13)
        q=np.zeros((len(z),len(r),9));v=np.zeros_like(q)
        q[:,:,2]=-1.;q[:,:,3]=1.;q[:,:,6]=1.
        source_values=np.zeros((len(z),len(r),3))
        source=StageRegularGaugeSource(
            source_values,np.zeros_like(source_values),z,r,
        )
        background={
            "wall_stiffness":0.,"v0":0.,"v1":0.,
            "beta_a":0.,"beta_b":0.,
            "wall_potential_a":0.,"wall_potential_b":0.,
        }
        rhs=NativeRegularSO3RHS(
            z,r,source,0.,background,np.zeros((2,len(r))),
            live_normal_wall_gauge=True,
            boundary_closure_mode="wall_owner_last_experimental",
        )
        acceleration,diagnostic=rhs.acceleration(
            0.,q,v,gauge_source_second_time=np.zeros_like(source_values),
            capture_boundary_stages=True,
        )
        normal=diagnostic["normal_wall_gauge"]
        self.assertTrue(diagnostic["finite"])
        self.assertTrue(normal["passed"])
        self.assertTrue(normal["coupled_block"]["passed"])
        self.assertLess(normal["final_residual"]["maximum"],1e-10)
        self.assertEqual(
            diagnostic["boundary_parameters"]["normal_wall_method"],
            "direct_coupled_4x4_both_walls",
        )
        self.assertEqual(diagnostic["boundary_stage_names"][-2:], [
            "post_wall_owner_reconciliation", "post_axis_operator_repair",
        ])
        for field in (
            "v_z=h_zr/r","d=(h_rr-h_perp)/r^2","v_0=h_0r/r",
        ):
            self.assertLess(
                diagnostic["axis_fit_preference_defect"]["by_field"][field],
                1e-12,
            )
        for wall in ("lower","upper"):
            record=wall_junction_second_tangent(
                q,v,acceleration,z,r,background,wall,
            )
            self.assertLess(np.max(np.abs(record["DX2J_tensor"])),1e-10)
            self.assertLess(np.max(np.abs(
                record["separate_rows"]["DX2_Phi_robin"]
            )),1e-11)
            self.assertLess(np.max(np.abs(
                record["separate_rows"]["DX2_chi_neumann"]
            )),1e-11)
        with mock.patch(
            "bhps.nonlinear_regular_so3_evolution."
            "compact_wall_normal_gauge_acceleration_residuals",
            return_value={"walls":[],"maximum":1e-10},
        ):
            with self.assertRaisesRegex(RuntimeError,"full unbuffered"):
                rhs.acceleration(
                    0.,q,v,
                    gauge_source_second_time=np.zeros_like(source_values),
                )

    def test_direct_coupled_phi_normal_block_closes_full_wall(self):
        z=np.linspace(1,2,17);r=np.linspace(0,1,25)
        zz,rr=np.meshgrid(z,r,indexing="ij")
        q=np.zeros((len(z),len(r),9));v=np.zeros_like(q);a=np.zeros_like(q)
        q[:,:,2]=-1+.01*zz;q[:,:,3]=1+.02*zz
        q[:,:,4]=.01;q[:,:,5]=.005
        q[:,:,6]=1.1+.02*zz+.01*rr**2
        q[:,:,7]=.03*zz+.01*rr**2;q[:,:,8]=.02*zz
        v[:,:,2]=.01*(1+zz);v[:,:,3]=-.02
        v[:,:,6]=.015;v[:,:,7]=.01;v[:,:,8]=.02
        for field in range(9):
            a[:,:,field]=(.03+.002*field)*(1+.1*zz+.02*rr**2)
        source=np.zeros((len(z),len(r),3));source_t=np.zeros_like(source)
        source_tt=np.zeros_like(source)
        background={
            "wall_stiffness":2.,"v0":.02,"v1":.06,
            "beta_a":.1,"beta_b":.12,
            "wall_potential_a":0.,"wall_potential_b":0.,
        }
        coupled,diagnostic=solve_compact_wall_coupled_phi_normal_acceleration(
            q,v,a,source,source_t,source_tt,z,r,background,
        )
        with self.assertRaisesRegex(RuntimeError,"linear-residual gate"):
            solve_compact_wall_coupled_phi_normal_acceleration(
                q,v,a,source,source_t,source_tt,z,r,background,
                maximum_normalized_linear_residual=1e-30,
            )
        normal=np.stack((coupled[0,:,6],coupled[-1,:,6]))
        solved,_=apply_compact_wall_acceleration(
            q,v,coupled,z,r,background,normal,fill_axis_after=False,
        )
        normal_residual=compact_wall_normal_gauge_acceleration_residuals(
            q,v,solved,source,source_t,source_tt,z,r,background,
            radial_buffer=0,
        )
        self.assertTrue(diagnostic["passed"])
        self.assertLess(diagnostic["maximum_condition"],2.)
        self.assertLess(normal_residual["maximum"],1e-12)
        for wall in ("lower","upper"):
            record=wall_junction_second_tangent(
                q,v,solved,z,r,background,wall,
            )
            self.assertLess(np.max(np.abs(
                record["separate_rows"]["DX2_Phi_robin"]
            )),1e-12)
            self.assertLess(np.max(np.abs(
                record["separate_rows"]["DX2_chi_neumann"]
            )),1e-12)

    def test_wall_owner_last_removes_axis_and_outer_stencil_defects(self):
        data=manufactured_state(nz=17,nr=25)
        q=data["position"];v=data["velocity"]
        z=data["z"];r=data["r"];background=data["background"]
        zz,rr=np.meshgrid(z,r,indexing="ij")
        a=np.zeros_like(q)
        for field in range(9):
            a[:,:,field]=(
                (.03+.004*field)*(1+.2*zz)
                *np.exp((.4+.03*field)*rr**2)
            )
        a=fill_regular_axis(a,r)
        normal=np.stack((a[0,:,6],a[-1,:,6]))

        wall_prefill,_=apply_compact_wall_acceleration(
            q,v,a,z,r,background,normal,fill_axis_after=False,
        )
        wall_then_axis,_=apply_compact_wall_acceleration(
            q,v,a,z,r,background,normal,
        )
        outer_after_wall,_=apply_outer_sommerfeld_acceleration(
            q,v,wall_then_axis,q,np.zeros_like(q),0.,r,
        )
        outer_first,_=apply_outer_sommerfeld_acceleration(
            q,v,a,q,np.zeros_like(q),0.,r,
        )
        owner_wall,_=apply_compact_wall_acceleration(
            q,v,outer_first,z,r,background,normal,fill_axis_after=False,
        )
        prefit=fill_regular_axis(owner_wall,r)
        self.assertGreater(np.max(np.abs(
            owner_wall[:,0,(1,4,5)]-prefit[:,0,(1,4,5)]
        )),1e-5)
        owner_last=reconcile_wall_owner_axis_null_channels(owner_wall,r)
        np.testing.assert_array_equal(
            owner_last[:,:,(0,2,3,6,7,8)],
            owner_wall[:,:,(0,2,3,6,7,8)],
        )
        np.testing.assert_array_equal(
            owner_last[:,1:,(1,4,5)],owner_wall[:,1:,(1,4,5)],
        )

        def endpoint_maximum(acceleration,index):
            return max(
                np.linalg.norm(wall_junction_second_tangent(
                    q,v,acceleration,z,r,background,wall,
                )["DX2J_tensor"][index])
                for wall in ("lower","upper")
            )

        self.assertLess(endpoint_maximum(wall_prefill,0),1e-11)
        self.assertGreater(endpoint_maximum(wall_then_axis,0),1e-11)
        self.assertGreater(endpoint_maximum(outer_after_wall,-1),1.)
        self.assertLess(endpoint_maximum(owner_last,0),1e-11)
        self.assertLess(endpoint_maximum(owner_last,-1),1e-11)
        np.testing.assert_array_equal(
            owner_last[1:-1,0][:,(0,2,3,6,7,8)],
            outer_first[1:-1,0][:,(0,2,3,6,7,8)],
        )

        audit=evaluate_boundary_stage_sequence(
            q,v,[
                {"name":"outer_first","acceleration":outer_first},
                {"name":"wall_owner_last","acceleration":owner_last},
            ],z,r,background,
        )
        self.assertTrue(audit["finite"])
        for wall in ("lower","upper"):
            jump=audit["jumps"][0]["walls"][wall]
            self.assertLess(jump["causal_identity_maximum_absolute_defect"],1e-14)
            self.assertEqual(jump["velocity_hessian_change_maximum_absolute"],0.)


if __name__ == "__main__":
    unittest.main()
