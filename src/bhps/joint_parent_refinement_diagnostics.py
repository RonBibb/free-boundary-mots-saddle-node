"""Frozen meshes and refinement diagnostics for Protocol 125.

This module is deliberately result-agnostic.  It defines the validation
coordinates and the correction/refinement functionals before either N0 or N1
is constructed.  It neither builds a parent nor writes an artifact.
"""

from __future__ import annotations

import numpy as np

from bhps.matched_staged_continuum import hash_arrays


PHYSICAL_COMPONENT_ORDER = (
    "h_z0", "h_zr", "h_00", "h_perp", "h_rr", "h_0r", "h_zz",
    "Phi", "chi",
)
HZZ_INDEX = PHYSICAL_COMPONENT_ORDER.index("h_zz")
AXIS_ACCELERATION_IMAGE_ORDER = (
    "drr_hrr_minus_hperp_from_q4",
    "dr_h0r_from_q5",
)
AXIS_IMAGE_SMALL_LIMIT = 0.05
AXIS_IMAGE_ORDER_ONE_LIMIT = 0.5
AXIS_IMAGE_DIFFERENCE_LIMIT = 2e-3
COLLAR_EDGES = np.asarray(
    (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf),
)
VALIDATION_MESH_SPECS = {
    "V0": (105, 235, "52d00574079a81fb84c23c74a76fe37f046a37aac59a396864994b983ad59ebf"),
    "V1": (129, 289, "9f25e8a580b640f3ef0799647929b07f9f6744d84d32bbcfe9150ce7cf3b8dd6"),
    "V2": (153, 343, "8e112ea841ccdf4e1cd3c932877f57d3e9046575d2c8cb0a978c735108186f8b"),
}
DENSE_WALL_SHA256 = (
    "8220cfeb81994fb820ed219f5fedb4c1f2ab8ae63798c86444649540323f90d6"
)
DENSE_OUTER_SHA256 = (
    "250c209adc013864fc8e36dc81c7a6ebe549531b025704f7e4e7182b31b1ae3c"
)


def frozen_validation_meshes():
    """Return fresh copies of every fixed Protocol-125 validation mesh."""
    meshes = {}
    for name, (nz, nr, expected) in VALIDATION_MESH_SPECS.items():
        z = np.linspace(1.0, np.e, nz)
        r = np.linspace(0.0, 12.0, nr)
        found = hash_arrays(z, r)
        if found != expected:
            raise RuntimeError(f"{name} coordinate digest differs from protocol")
        meshes[name] = {"z": z, "r": r, "sha256": found}
    dense_r = np.linspace(0.0, 12.0, 1025)
    dense_z = np.linspace(1.0, np.e, 1025)
    if hash_arrays(dense_r) != DENSE_WALL_SHA256:
        raise RuntimeError("dense-wall coordinate digest differs from protocol")
    if hash_arrays(dense_z) != DENSE_OUTER_SHA256:
        raise RuntimeError("dense-outer coordinate digest differs from protocol")
    meshes["dense_wall"] = {
        "r": dense_r, "sha256": DENSE_WALL_SHA256,
    }
    meshes["dense_outer"] = {
        "z": dense_z, "sha256": DENSE_OUTER_SHA256,
    }
    return meshes


def reduced_to_physical(values, r):
    """Map native reduced values to physical coordinate components."""
    q = np.asarray(values, dtype=float)
    r = np.asarray(r, dtype=float)
    if q.ndim < 2 or q.shape[-2:] != (len(r), 9):
        raise ValueError("reduced field has the wrong radial/component shape")
    radius = r.reshape((1,)*(q.ndim-2)+(len(r),))
    return np.stack((
        q[..., 0], radius*q[..., 1], q[..., 2], q[..., 3],
        q[..., 3]+radius**2*q[..., 4], radius*q[..., 5],
        q[..., 6], q[..., 7], q[..., 8],
    ), axis=-1)


def _wall_rows(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 9 or values.shape[0] < 2:
        raise ValueError("wall diagnostic field must have shape (z,r,9)")
    return values[[0, -1]]


def _nodal_widths(r):
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or len(r) < 2 or np.any(np.diff(r) <= 0.0):
        raise ValueError("radial coordinate must be strictly increasing")
    widths = np.empty_like(r)
    widths[0] = 0.5*(r[1]-r[0])
    widths[-1] = 0.5*(r[-1]-r[-2])
    widths[1:-1] = 0.5*(r[2:]-r[:-2])
    return widths


def _proper_geometry(position, r):
    wall = reduced_to_physical(_wall_rows(position), r)
    h_perp = wall[:, :, PHYSICAL_COMPONENT_ORDER.index("h_perp")]
    h_rr = wall[:, :, PHYSICAL_COMPONENT_ORDER.index("h_rr")]
    if np.any(h_perp <= 0.0) or np.any(h_rr <= 0.0):
        raise RuntimeError("proper-wall metric is not positive")
    increments = (
        0.5*(np.sqrt(h_rr[:, 1:])+np.sqrt(h_rr[:, :-1]))
        * np.diff(r)[None, :]
    )
    proper_radius = np.concatenate((
        np.zeros((2, 1)), np.cumsum(increments, axis=1),
    ), axis=1)
    weights = (
        4.0*np.pi*r[None, :]**2*h_perp*np.sqrt(h_rr)
        * _nodal_widths(r)[None, :]
    )
    return wall, proper_radius, weights


def _localization(profile, proper_radius, weights):
    profile = np.asarray(profile, dtype=float)
    proper_radius = np.asarray(proper_radius, dtype=float)
    weights = np.asarray(weights, dtype=float)
    energy = weights*profile**2
    total = float(np.sum(energy))
    fractions = []
    for lower, upper in zip(COLLAR_EDGES[:-1], COLLAR_EDGES[1:]):
        mask = proper_radius >= lower
        if np.isfinite(upper):
            mask &= proper_radius < upper
        fractions.append(float(np.sum(energy[mask])/total) if total > 0.0 else 0.0)
    if total > 0.0:
        index = int(np.searchsorted(np.cumsum(energy), 0.9*total))
        radius_90 = float(proper_radius[min(index, len(proper_radius)-1)])
    else:
        radius_90 = 0.0
    integral2 = float(np.trapezoid(profile**2, proper_radius))
    integral4 = float(np.trapezoid(profile**4, proper_radius))
    support = integral2**2/integral4 if integral4 > 0.0 else 0.0
    return {
        "collar_edges": COLLAR_EDGES.tolist(),
        "area_energy_fractions": fractions,
        "area_energy_fraction_sum": float(sum(fractions)),
        "proper_radius_90": radius_90,
        "effective_proper_support_length": float(support),
        "effective_support_quadrature": (
            "(trapezoid(c_hzz^2,dell)^2)/trapezoid(c_hzz^4,dell)"
        ),
    }


def correction_profile(
    position, bulk_acceleration, compatible_acceleration, r, *,
    require_frozen_mesh=True,
):
    """Evaluate one parent's signed physical correction on both walls."""
    r = np.asarray(r, dtype=float)
    if bool(require_frozen_mesh) and hash_arrays(r) != DENSE_WALL_SHA256:
        raise ValueError("correction profile requires the frozen 1025-point wall mesh")
    bulk = reduced_to_physical(_wall_rows(bulk_acceleration), r)
    compatible = reduced_to_physical(_wall_rows(compatible_acceleration), r)
    _, proper_radius, weights = _proper_geometry(position, r)
    denominator = np.maximum.reduce((
        np.ones_like(bulk), np.abs(bulk), np.abs(compatible),
    ))
    correction = (compatible-bulk)/denominator
    if not all(np.all(np.isfinite(item)) for item in (
        correction, proper_radius, weights,
    )):
        raise RuntimeError("correction profile is nonfinite")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise RuntimeError("proper-wall quadrature has zero measure")
    hzz = correction[:, :, HZZ_INDEX]
    full_weighted_rms = float(np.sqrt(
        np.sum(weights[:, :, None]*correction**2)
        /(len(PHYSICAL_COMPONENT_ORDER)*total_weight)
    ))
    hzz_weighted_rms = float(np.sqrt(
        np.sum(weights*hzz**2)/total_weight
    ))
    local = {
        wall: _localization(hzz[index], proper_radius[index], weights[index])
        for index, wall in enumerate(("lower", "upper"))
    }
    return {
        "physical_component_order": PHYSICAL_COMPONENT_ORDER,
        "signed_normalized_correction": correction,
        "proper_radius": proper_radius,
        "proper_wall_weights": weights,
        "full_physical_Linf": float(np.max(np.abs(correction))),
        "full_physical_weighted_RMS": full_weighted_rms,
        "hzz_C": float(np.max(np.abs(hzz))),
        "hzz_W": hzz_weighted_rms,
        "localization": local,
        "small_full_Linf_gate": bool(np.max(np.abs(correction)) <= 0.05),
        "small_full_RMS_gate": bool(full_weighted_rms <= 0.01),
        "order_one_failure": bool(np.max(np.abs(correction)) > 0.5),
    }


def _decreases_or_both_floor(coarse, refined, floor=1e-12):
    return bool(
        (coarse <= floor and refined <= floor)
        or refined < coarse
    )


def axis_acceleration_derivative_image_profile(
    bulk_acceleration,
    compatible_acceleration,
    z,
    r=None,
    *,
    require_frozen_mesh=True,
):
    """Score the two analytic acceleration images on the compact axis.

    Inputs may be direct axis evaluations ``(nz,9)`` or full evaluations
    ``(nz,nr,9)`` with an explicit radial coordinate whose first node is exact
    IEEE positive zero.  No stored correction profile is interpolated.
    """
    z = np.asarray(z, dtype=float)
    bulk = np.asarray(bulk_acceleration, dtype=float)
    compatible = np.asarray(compatible_acceleration, dtype=float)
    if z.ndim != 1 or len(z) < 2 or np.any(np.diff(z) <= 0.0):
        raise ValueError("axis-image compact coordinate must be increasing")
    if bool(require_frozen_mesh):
        frozen_z = frozen_validation_meshes()["V2"]["z"]
        if not np.array_equal(z, frozen_z):
            raise ValueError("axis-image gate requires the frozen V2 compact mesh")
    if bulk.shape != compatible.shape:
        raise ValueError("bulk/compatible axis-image accelerations differ in shape")
    if bulk.ndim == 2:
        if bulk.shape != (len(z), 9) or r is not None:
            raise ValueError("direct axis acceleration must have shape (nz,9)")
        bulk_axis = bulk
        compatible_axis = compatible
        input_form = "direct-continuous-axis"
    elif bulk.ndim == 3:
        r = np.asarray(r, dtype=float)
        if (
            r.ndim != 1
            or bulk.shape != (len(z), len(r), 9)
            or len(r) < 1
            or r[0] != 0.0
            or np.signbit(r[0])
            or np.any(np.diff(r) <= 0.0)
        ):
            raise ValueError("full axis-image evaluation requires exact +0 radial axis")
        bulk_axis = bulk[:, 0]
        compatible_axis = compatible[:, 0]
        input_form = "full-continuous-mesh-axis-column"
    else:
        raise ValueError("axis-image acceleration must be rank two or three")
    if not all(np.all(np.isfinite(value)) for value in (
        z, bulk_axis, compatible_axis,
    )):
        raise ValueError("axis-image inputs must be finite")

    bulk_images = np.stack((2.0*bulk_axis[:, 4], bulk_axis[:, 5]))
    compatible_images = np.stack((
        2.0*compatible_axis[:, 4], compatible_axis[:, 5],
    ))
    denominator = np.maximum.reduce((
        np.ones_like(bulk_images),
        np.abs(bulk_images),
        np.abs(compatible_images),
    ))
    correction = (compatible_images-bulk_images)/denominator
    records = {}
    for image_index, name in enumerate(AXIS_ACCELERATION_IMAGE_ORDER):
        profile = correction[image_index]
        maximum_index = int(np.argmax(np.abs(profile)))
        maximum = float(np.abs(profile[maximum_index]))
        records[name] = {
            "bulk_image": bulk_images[image_index].copy(),
            "compatible_image": compatible_images[image_index].copy(),
            "signed_normalized_correction": profile.copy(),
            "K": maximum,
            "maximum_index": maximum_index,
            "maximum_z": float(z[maximum_index]),
            "small_image_gate": bool(maximum <= AXIS_IMAGE_SMALL_LIMIT),
            "order_one_failure": bool(maximum > AXIS_IMAGE_ORDER_ONE_LIMIT),
        }
    return {
        "method": "Protocol-125-analytic-axis-acceleration-derivative-images",
        "image_order": AXIS_ACCELERATION_IMAGE_ORDER,
        "z": z.copy(),
        "z_sha256": hash_arrays(z),
        "input_form": input_form,
        "raw_q4_bulk": bulk_axis[:, 4].copy(),
        "raw_q4_compatible": compatible_axis[:, 4].copy(),
        "raw_q4_change": (compatible_axis[:, 4]-bulk_axis[:, 4]).copy(),
        "raw_q5_bulk": bulk_axis[:, 5].copy(),
        "raw_q5_compatible": compatible_axis[:, 5].copy(),
        "raw_q5_change": (compatible_axis[:, 5]-bulk_axis[:, 5]).copy(),
        "images": records,
        "small_limit": AXIS_IMAGE_SMALL_LIMIT,
        "order_one_limit": AXIS_IMAGE_ORDER_ONE_LIMIT,
        "all_small": bool(all(
            record["small_image_gate"] for record in records.values()
        )),
        "any_order_one": bool(any(
            record["order_one_failure"] for record in records.values()
        )),
    }


def compare_axis_acceleration_image_refinement(n0, n1):
    """Apply strict-decrease, difference, and envelope gates to both images."""
    if tuple(n0.get("image_order", ())) != AXIS_ACCELERATION_IMAGE_ORDER:
        raise ValueError("N0 acceleration-image order changed")
    if tuple(n1.get("image_order", ())) != AXIS_ACCELERATION_IMAGE_ORDER:
        raise ValueError("N1 acceleration-image order changed")
    z0 = np.asarray(n0.get("z"), dtype=float)
    z1 = np.asarray(n1.get("z"), dtype=float)
    frozen_z = frozen_validation_meshes()["V2"]["z"]
    if not (
        np.array_equal(z0, frozen_z)
        and np.array_equal(z1, frozen_z)
        and str(n0.get("z_sha256")) == hash_arrays(frozen_z)
        and str(n1.get("z_sha256")) == hash_arrays(frozen_z)
    ):
        raise ValueError("axis-image refinement requires the common frozen V2 axis")
    records = {}
    for name in AXIS_ACCELERATION_IMAGE_ORDER:
        try:
            coarse_record = n0["images"][name]
            refined_record = n1["images"][name]
            coarse = np.asarray(
                coarse_record["signed_normalized_correction"], dtype=float,
            )
            refined = np.asarray(
                refined_record["signed_normalized_correction"], dtype=float,
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"axis-image record {name} is incomplete") from error
        if coarse.shape != z0.shape or refined.shape != z0.shape or not all(
            np.all(np.isfinite(value)) for value in (coarse, refined)
        ):
            raise ValueError(f"axis-image profile {name} is invalid")
        coarse_k = float(np.max(np.abs(coarse)))
        refined_k = float(np.max(np.abs(refined)))
        if not (
            np.isclose(coarse_k, float(coarse_record["K"]), rtol=0.0, atol=0.0)
            and np.isclose(
                refined_k, float(refined_record["K"]), rtol=0.0, atol=0.0,
            )
        ):
            raise ValueError(f"axis-image stored maximum {name} differs")
        difference = refined-coarse
        maximum_index = int(np.argmax(np.abs(difference)))
        d_linf = float(np.abs(difference[maximum_index]))
        gates = {
            "N0_small": bool(coarse_k <= AXIS_IMAGE_SMALL_LIMIT),
            "N1_small": bool(refined_k <= AXIS_IMAGE_SMALL_LIMIT),
            "N0_not_order_one": bool(coarse_k <= AXIS_IMAGE_ORDER_ONE_LIMIT),
            "N1_not_order_one": bool(refined_k <= AXIS_IMAGE_ORDER_ONE_LIMIT),
            "strict_decrease": _decreases_or_both_floor(coarse_k, refined_k),
            "profile_difference": bool(
                d_linf <= AXIS_IMAGE_DIFFERENCE_LIMIT
            ),
            "conservative_envelope": bool(
                refined_k+d_linf <= AXIS_IMAGE_SMALL_LIMIT
            ),
        }
        records[name] = {
            "K_N0": coarse_k,
            "K_N1": refined_k,
            "signed_profile_difference_N1_minus_N0": difference.copy(),
            "D": d_linf,
            "maximum_difference_index": maximum_index,
            "maximum_difference_z": float(z0[maximum_index]),
            "gates": gates,
            "pass": bool(all(gates.values())),
        }
    return {
        "image_order": AXIS_ACCELERATION_IMAGE_ORDER,
        "images": records,
        "difference_limit": AXIS_IMAGE_DIFFERENCE_LIMIT,
        "small_limit": AXIS_IMAGE_SMALL_LIMIT,
        "order_one_limit": AXIS_IMAGE_ORDER_ONE_LIMIT,
        "pass": bool(all(record["pass"] for record in records.values())),
    }


def compare_correction_refinement(
    n0, n1, position_v2_n0, position_v2_n1, r_v2,
):
    """Apply only the frozen N0/N1 hzz correction/localization subgates."""
    c0 = np.asarray(n0["signed_normalized_correction"], dtype=float)
    c1 = np.asarray(n1["signed_normalized_correction"], dtype=float)
    w0 = np.asarray(n0["proper_wall_weights"], dtype=float)
    w1 = np.asarray(n1["proper_wall_weights"], dtype=float)
    spacing = maximum_v2_proper_spacing(
        position_v2_n0, position_v2_n1, r_v2,
    )
    if c0.shape != c1.shape or w0.shape != w1.shape or spacing.shape != (2,):
        raise ValueError("refinement profiles must share one dense wall mesh")
    difference = c1[:, :, HZZ_INDEX]-c0[:, :, HZZ_INDEX]
    mean_weight = 0.5*(w0+w1)
    denominator = float(np.sum(mean_weight))
    if denominator <= 0.0:
        raise RuntimeError("mean refinement measure is zero")
    d_linf = float(np.max(np.abs(difference)))
    d_rms = float(np.sqrt(np.sum(mean_weight*difference**2)/denominator))
    localization = {}
    localization_gate = True
    if float(n0["hzz_C"]) > 1e-12:
        for index, wall in enumerate(("lower", "upper")):
            coarse = n0["localization"][wall]
            refined = n1["localization"][wall]
            r90_pass = bool(
                refined["proper_radius_90"]+spacing[index]
                >= coarse["proper_radius_90"]
            )
            support_pass = bool(
                refined["effective_proper_support_length"]+spacing[index]
                >= coarse["effective_proper_support_length"]
            )
            localization[wall] = {
                "maximum_adjacent_V2_proper_spacing": float(spacing[index]),
                "proper_radius_90_nonconcentration": r90_pass,
                "effective_support_nonconcentration": support_pass,
            }
            localization_gate &= r90_pass and support_pass
    else:
        localization = {
            wall: {
                "maximum_adjacent_V2_proper_spacing": float(spacing[index]),
                "proper_radius_90_nonconcentration": True,
                "effective_support_nonconcentration": True,
            }
            for index, wall in enumerate(("lower", "upper"))
        }
    gates = {
        "C_decreases": _decreases_or_both_floor(n0["hzz_C"], n1["hzz_C"]),
        "W_decreases": _decreases_or_both_floor(n0["hzz_W"], n1["hzz_W"]),
        "profile_difference": bool(d_linf <= 2e-3),
        "C_envelope": bool(float(n1["hzz_C"])+d_linf <= 0.05),
        "W_envelope": bool(float(n1["hzz_W"])+d_rms <= 0.01),
        "localization": bool(localization_gate),
    }
    return {
        "D_Linf": d_linf,
        "D_RMS": d_rms,
        "gates": gates,
        "hzz_refinement_pass": bool(all(gates.values())),
        "localization": localization,
    }


def maximum_v2_proper_spacing(
    position_n0, position_n1, r_v2, *, require_frozen_mesh=True,
):
    """Return the wallwise largest spacing of the mean V2 proper radius."""
    r_v2 = np.asarray(r_v2, dtype=float)
    if bool(require_frozen_mesh):
        frozen_r = frozen_validation_meshes()["V2"]["r"]
        if not np.array_equal(r_v2, frozen_r):
            raise ValueError("proper-spacing gate requires the frozen V2 radial mesh")
    _, proper0, _ = _proper_geometry(position_n0, r_v2)
    _, proper1, _ = _proper_geometry(position_n1, r_v2)
    mean_proper = 0.5*(proper0+proper1)
    return np.max(np.diff(mean_proper, axis=1), axis=1)


def _scaled_linf(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or not all(np.all(np.isfinite(item)) for item in (
        left, right,
    )):
        raise ValueError("scaled comparison arrays must be finite and shape matched")
    return float(np.max(
        np.abs(left-right)/np.maximum.reduce((
            np.ones_like(left), np.abs(left), np.abs(right),
        ))
    ))


def adjudicate_correction_refinement(
    n0,
    n1,
    position_v2_n0,
    position_v2_n1,
    r_v2,
    *,
    hzz_zz_n0,
    hzz_zz_n1,
    a_hzz_n0,
    a_hzz_n1,
    axis_image_n0,
    axis_image_n1,
):
    """Combine every correction-size and two-parent refinement gate."""
    refinement = compare_correction_refinement(
        n0, n1, position_v2_n0, position_v2_n1, r_v2,
    )
    hzz_zz_difference = _scaled_linf(hzz_zz_n0, hzz_zz_n1)
    a_hzz_difference = _scaled_linf(a_hzz_n0, a_hzz_n1)
    axis_images = compare_axis_acceleration_image_refinement(
        axis_image_n0, axis_image_n1,
    )
    gates = {
        "N0_full_physical_Linf": bool(n0["small_full_Linf_gate"]),
        "N0_full_physical_RMS": bool(n0["small_full_RMS_gate"]),
        "N0_not_order_one": not bool(n0["order_one_failure"]),
        "N1_full_physical_Linf": bool(n1["small_full_Linf_gate"]),
        "N1_full_physical_RMS": bool(n1["small_full_RMS_gate"]),
        "N1_not_order_one": not bool(n1["order_one_failure"]),
        "hzz_refinement": bool(refinement["hzz_refinement_pass"]),
        "hzz_zz_difference": bool(hzz_zz_difference <= 2e-3),
        "a_hzz_difference": bool(a_hzz_difference <= 2e-3),
        "q4_q5_axis_acceleration_images": bool(axis_images["pass"]),
    }
    return {
        "refinement": refinement,
        "hzz_zz_scaled_Linf_difference": hzz_zz_difference,
        "a_hzz_scaled_Linf_difference": a_hzz_difference,
        "axis_acceleration_derivative_images": axis_images,
        "gates": gates,
        "pass": bool(all(gates.values())),
    }
