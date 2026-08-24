import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.lapse_acceleration_corner import construct_lapse_acceleration_completion,construct_localized_target_lapse_acceleration_completion,construct_minimum_norm_lapse_acceleration_completion,construct_projected_target_lapse_acceleration_completion,construct_target_relative_lapse_acceleration_completion,time_time_israel_second_corner_fields


class LapseAccelerationCornerTests(unittest.TestCase):
    def test_cubic_completion_clears_both_rows(self):
        z=np.linspace(1,np.e,17);r=np.linspace(0,3,21);shape=(len(z),len(r))
        psi=np.exp(-.2*(z[:,None]-1))*np.ones((1,len(r)));alpha=psi.copy()
        a=np.zeros(shape);phi=.1+.02*np.cos(r)[None,:]*np.ones((len(z),1))
        scalar_acc=.03*np.sin(r)[None,:]*np.ones((len(z),1))
        acceleration={
            "zz":.2+.1*np.cos(r)[None,:]*np.ones((len(z),1)),
            "Dz":derivative_matrix(z,1,7),
        }
        background={
            "wall_stiffness":20.,"v0":.11,"v1":.09,
            "beta_a":1.,"beta_b":.8,"wall_potential_a":0.,"wall_potential_b":0.,
        }
        baseline=time_time_israel_second_corner_fields(
            acceleration,alpha,psi,a,phi,background,scalar_acc,None,3,
        )
        self.assertGreater(max(np.max(np.abs(w["residual"])) for w in baseline["walls"]),1e-3)
        completion=construct_lapse_acceleration_completion(
            z,acceleration,alpha,psi,a,phi,background,scalar_acc,
        )
        final=time_time_israel_second_corner_fields(
            acceleration,alpha,psi,a,phi,background,scalar_acc,
            completion["lapse_acceleration"],3,
        )
        self.assertLess(max(np.max(np.abs(w["residual"])) for w in final["walls"]),2e-11)
        optimized=construct_minimum_norm_lapse_acceleration_completion(
            z,acceleration,alpha,psi,a,phi,background,scalar_acc,
        )
        optimized_fields=time_time_israel_second_corner_fields(
            acceleration,alpha,psi,a,phi,background,scalar_acc,
            optimized["lapse_acceleration"],3,
        )
        self.assertLess(max(np.max(np.abs(w["residual"])) for w in optimized_fields["walls"]),2e-11)
        self.assertLess(
            optimized["maximum_relative_lapse_acceleration"],
            completion["maximum_relative_lapse_acceleration"],
        )
        target=np.full_like(alpha,.37)
        targeted=construct_target_relative_lapse_acceleration_completion(
            z,acceleration,alpha,psi,a,phi,background,scalar_acc,target,
        )
        targeted_fields=time_time_israel_second_corner_fields(
            acceleration,alpha,psi,a,phi,background,scalar_acc,
            targeted["lapse_acceleration"],3,
        )
        self.assertLess(max(
            np.max(np.abs(wall["residual"])/wall["scale"])
            for wall in targeted_fields["walls"]
        ),1e-11)
        self.assertLess(
            np.sqrt(np.mean(targeted["relative_acceleration_mismatch"]**2)),
            np.sqrt(np.mean((optimized["lapse_acceleration"]/alpha-target)**2)),
        )
        projected=construct_projected_target_lapse_acceleration_completion(
            z,acceleration,alpha,psi,a,phi,background,scalar_acc,target,
        )
        projected_fields=time_time_israel_second_corner_fields(
            acceleration,alpha,psi,a,phi,background,scalar_acc,
            projected["lapse_acceleration"],3,
        )
        self.assertLess(max(
            np.max(np.abs(wall["residual"])/wall["scale"])
            for wall in projected_fields["walls"]
        ),1e-11)
        self.assertLess(
            np.sqrt(np.mean(projected["relative_acceleration_mismatch"]**2)),
            np.sqrt(np.mean(targeted["relative_acceleration_mismatch"]**2)),
        )
        localized=construct_localized_target_lapse_acceleration_completion(
            z,acceleration,alpha,psi,a,phi,background,scalar_acc,target,.2,
        )
        localized_fields=time_time_israel_second_corner_fields(
            acceleration,alpha,psi,a,phi,background,scalar_acc,
            localized["lapse_acceleration"],3,
        )
        self.assertLess(max(
            np.max(np.abs(wall["residual"])/wall["scale"])
            for wall in localized_fields["walls"]
        ),1e-11)
        self.assertEqual(localized["logarithmic_width"],.2)


if __name__=="__main__":unittest.main()
