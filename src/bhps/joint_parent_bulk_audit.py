"""Result-independent open-bulk constraint diagnostics for Protocol 125.

The functions in this module evaluate only the Hamiltonian and stationary-Phi
bulk equations.  They never replace a compact-wall or radial-outer row.  The
source-node lane uses the frozen seven-point polynomial operators; the
off-node lane consumes analytic jets from the candidate and reference
representations.

Both lanes use the same pointwise normalization,

``max(1, abs(raw) + abs(reference_defect))``,

and use the candidate collapse-scalar gradients in both the raw and reference
defect equations.  The lapse is deliberately absent: the spatial conformal
factor is reconstructed only from ``h_zz*h_rr*h_perp**2``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from bhps.anisotropic_geometry import axisymmetric_diagonal_geometry
from bhps.anisotropic_initial_data import _shape_operators
from bhps.gw_slice_high_order_solver import derivative_matrix


SOURCE_STENCIL_WIDTH = 7
EQUATION_ORDER = ("hamiltonian", "Phi")
JET_LANES = ("value", "z", "r", "zz", "rr")


def _validated_coordinates(z, r, *, require_source_stencil):
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if z.ndim != 1 or r.ndim != 1:
        raise ValueError("bulk-audit coordinates must be one-dimensional")
    if not len(z) or not len(r):
        raise ValueError("bulk-audit grids must be nonempty")
    if require_source_stencil and (
        len(z) < SOURCE_STENCIL_WIDTH or len(r) < SOURCE_STENCIL_WIDTH
    ):
        raise ValueError("bulk-audit grids must support the seven-point stencil")
    if (
        np.any(~np.isfinite(z))
        or np.any(~np.isfinite(r))
        or np.any(np.diff(z) <= 0.0)
        or np.any(np.diff(r) <= 0.0)
        or np.any(r < 0.0)
    ):
        raise ValueError("bulk-audit coordinates must be finite and increasing")
    return z, r


def _result(
    raw, defect, *, method, axis_treatment, source_stencil_width=None,
):
    raw = {
        name: np.asarray(raw[name], dtype=float)
        for name in EQUATION_ORDER
    }
    defect = {
        name: np.asarray(defect[name], dtype=float)
        for name in EQUATION_ORDER
    }
    if any(raw[name].shape != defect[name].shape for name in EQUATION_ORDER):
        raise ValueError("raw and reference-defect equation shapes differ")
    if not all(
        np.all(np.isfinite(value))
        for collection in (raw, defect)
        for value in collection.values()
    ):
        raise RuntimeError("open-bulk equation evaluation is nonfinite")
    balanced = {
        name: raw[name]-defect[name]
        for name in EQUATION_ORDER
    }
    denominator = {
        name: np.maximum(1.0, np.abs(raw[name])+np.abs(defect[name]))
        for name in EQUATION_ORDER
    }
    raw_normalized = {
        name: np.abs(raw[name])/denominator[name]
        for name in EQUATION_ORDER
    }
    balanced_normalized = {
        name: np.abs(balanced[name])/denominator[name]
        for name in EQUATION_ORDER
    }
    reassembly = {
        name: balanced[name]-(raw[name]-defect[name])
        for name in EQUATION_ORDER
    }
    return {
        "equation_order": EQUATION_ORDER,
        "raw": raw,
        "defect": defect,
        "balanced": balanced,
        "common_denominator": denominator,
        "raw_normalized": raw_normalized,
        "balanced_normalized": balanced_normalized,
        "reassembly_defect": reassembly,
        "reassembly_Linf": float(max(
            np.max(np.abs(reassembly[name])) for name in EQUATION_ORDER
        )),
        "method": str(method),
        "axis_treatment": str(axis_treatment),
        "source_stencil_width": source_stencil_width,
        "normalization": "max(1,abs(raw)+abs(reference_defect))",
        "lapse_used": False,
        "candidate_chi_jets_reused_in_defect": True,
    }


def _fd_raw_equations(
    q, phi, z, r, a, b, c, background, chi_r, chi_z,
):
    """Evaluate the established shaped equations before row replacement."""
    shape = (len(z), len(r))
    q = np.asarray(q, dtype=float).reshape(shape)
    phi = np.asarray(phi, dtype=float).reshape(shape)
    a = np.asarray(a, dtype=float).reshape(shape)
    b = np.asarray(b, dtype=float).reshape(shape)
    c = np.asarray(c, dtype=float).reshape(shape)
    chi_r = np.asarray(chi_r, dtype=float).reshape(shape)
    chi_z = np.asarray(chi_z, dtype=float).reshape(shape)
    operators = _shape_operators(z, r, a, b, SOURCE_STENCIL_WIDTH)
    dz, dr, lap = (
        operators[name] for name in ("Dz", "Dr", "Lap")
    )
    qv = q.ravel()
    phiv = phi.ravel()
    psi = 1.0/(z[:, None]+q)
    if np.any(psi <= 0.0) or np.any(~np.isfinite(psi)):
        raise ValueError("bulk-audit conformal factor must be positive")
    psiv = psi.ravel()
    phi_z = dz @ phiv
    phi_r = dr @ phiv
    inverse_a = operators["inverse_a"]
    inverse_b = operators["inverse_b"]
    gradient = (
        inverse_a*(phi_z**2+chi_z.ravel()**2)
        + inverse_b*(phi_r**2+chi_r.ravel()**2)
    )
    scalar_bar = axisymmetric_diagonal_geometry(
        z,
        r,
        np.ones_like(psi),
        a,
        b,
        c,
        SOURCE_STENCIL_WIDTH,
    )["scalar_curvature"].ravel()
    mass = float(background["mass_squared"])
    potential = mass*phiv**2
    hamiltonian = (
        -6.0*(lap @ psiv)
        +(scalar_bar-gradient)*psiv
        -(-12.0+potential)*psiv**3
    )
    w_z = dz @ np.log(psiv)
    w_r = dr @ np.log(psiv)
    scalar = (
        lap @ phiv
        +3.0*(inverse_a*w_z*phi_z+inverse_b*w_r*phi_r)
        -mass*psiv**2*phiv
    )
    return {
        "hamiltonian": hamiltonian.reshape(shape),
        "Phi": scalar.reshape(shape),
    }


def open_anisotropic_bulk_terms_fd(
    position,
    z,
    r,
    reference_q,
    reference_phi,
    background,
):
    """Evaluate source-node open bulk terms with frozen seven-point operators.

    ``position`` is the completed nine-channel native position.  Only its
    spatial diagonal metric, Phi, and chi channels are read.  Returned arrays
    cover the complete rectangular mesh, but contain bulk formulas everywhere;
    the caller owns all face/collar masks and no boundary equation is inserted.
    """
    z, r = _validated_coordinates(z, r, require_source_stencil=True)
    position = np.asarray(position, dtype=float)
    shape = (len(z), len(r))
    if position.shape != (*shape, 9):
        raise ValueError("completed native position has the wrong shape")
    reference_q = np.asarray(reference_q, dtype=float)
    reference_phi = np.asarray(reference_phi, dtype=float)
    if reference_q.shape != shape or reference_phi.shape != shape:
        raise ValueError("finite-wall reference fields have the wrong shape")
    if not all(np.all(np.isfinite(value)) for value in (
        position, reference_q, reference_phi,
    )):
        raise ValueError("bulk-audit source arrays must be finite")

    radius = r[None, :]
    h_perp = position[:, :, 3]
    h_rr = h_perp+radius**2*position[:, :, 4]
    h_zz = position[:, :, 6]
    if np.any(np.stack((h_perp, h_rr, h_zz)) <= 0.0):
        raise ValueError("spatial determinant metric must be positive")
    psi = (h_zz*h_rr*h_perp**2)**0.125
    a = 0.5*np.log(h_zz/psi**2)
    b = 0.5*np.log(h_rr/psi**2)
    c = 0.5*np.log(h_perp/psi**2)
    candidate_q = 1.0/psi-z[:, None]
    phi = position[:, :, 7]
    chi = position[:, :, 8]
    dz = derivative_matrix(z, 1, SOURCE_STENCIL_WIDTH)
    dr = derivative_matrix(r, 1, SOURCE_STENCIL_WIDTH)
    chi_z = dz @ chi
    chi_r = (dr @ chi.T).T
    if r[0] == 0.0:
        chi_r[:, 0] = 0.0

    raw = _fd_raw_equations(
        candidate_q, phi, z, r, a, b, c, background, chi_r, chi_z,
    )
    zeros = np.zeros(shape)
    defect = _fd_raw_equations(
        reference_q,
        reference_phi,
        z,
        r,
        zeros,
        zeros,
        zeros,
        background,
        chi_r,
        chi_z,
    )
    return _result(
        raw,
        defect,
        method="source_node_polynomial_fd7_open_bulk",
        axis_treatment=(
            "native_seven_point_regular_axis" if r[0] == 0.0
            else "off_axis_only"
        ),
        source_stencil_width=SOURCE_STENCIL_WIDTH,
    )


def _extract_jet(mapping, names):
    if not isinstance(mapping, Mapping):
        raise ValueError("analytic jets must be supplied as a mapping")
    selected = next((name for name in names if name in mapping), None)
    if selected is None:
        raise ValueError(f"analytic jet is missing field {names[0]}")
    value = mapping[selected]
    if isinstance(value, Mapping):
        if any(lane not in value for lane in JET_LANES):
            raise ValueError(f"nested analytic jet {selected} is incomplete")
        return {
            lane: np.asarray(value[lane], dtype=float)
            for lane in JET_LANES
        }
    result = {"value": np.asarray(value, dtype=float)}
    for lane in JET_LANES[1:]:
        key = next(
            (
                f"{name}_{lane}" for name in names
                if f"{name}_{lane}" in mapping
            ),
            None,
        )
        if key is None:
            raise ValueError(f"analytic jet {selected} is missing {lane}")
        result[lane] = np.asarray(mapping[key], dtype=float)
    return result


def _validate_jet_shapes(fields, shape):
    for name, field in fields.items():
        for lane in JET_LANES:
            value = np.asarray(field[lane], dtype=float)
            if value.shape != shape:
                raise ValueError(f"analytic {name} {lane} has the wrong shape")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"analytic {name} {lane} is nonfinite")


def _log_jet(field):
    value = field["value"]
    if np.any(value <= 0.0):
        raise ValueError("spatial metric jet values must be positive")
    result = {"value": 0.5*np.log(value)}
    for direction in ("z", "r"):
        result[direction] = 0.5*field[direction]/value
        result[direction*2] = 0.5*(
            field[direction*2]/value
            -(field[direction]/value)**2
        )
    return result


def _combine_jets(*weighted):
    return {
        lane: sum(weight*field[lane] for weight, field in weighted)
        for lane in JET_LANES
    }


def _candidate_primitive_jets(candidate):
    log_z = _log_jet(candidate["h_zz"])
    log_r = _log_jet(candidate["h_rr"])
    log_p = _log_jet(candidate["h_perp"])
    w = _combine_jets((0.25, log_z), (0.25, log_r), (0.5, log_p))
    a = _combine_jets((1.0, log_z), (-1.0, w))
    b = _combine_jets((1.0, log_r), (-1.0, w))
    c = _combine_jets((1.0, log_p), (-1.0, w))
    psi = {
        "value": np.exp(w["value"]),
        "z": None,
        "r": None,
        "zz": None,
        "rr": None,
    }
    for direction in ("z", "r"):
        psi[direction] = psi["value"]*w[direction]
        psi[direction*2] = psi["value"]*(
            w[direction*2]+w[direction]**2
        )
    return {
        "psi": psi,
        "w": w,
        "a": a,
        "b": b,
        "c": c,
        "Phi": candidate["Phi"],
        "chi": candidate["chi"],
    }


def _reference_primitive_jets(reference, z):
    q = reference["q"]
    s = z[:, None]+q["value"]
    if np.any(s <= 0.0):
        raise ValueError("reference z+q must remain positive")
    psi = {"value": 1.0/s}
    s_first = {"z": 1.0+q["z"], "r": q["r"]}
    for direction in ("z", "r"):
        psi[direction] = -psi["value"]**2*s_first[direction]
        psi[direction*2] = (
            2.0*psi["value"]**3*s_first[direction]**2
            -psi["value"]**2*q[direction*2]
        )
    w = {
        "value": np.log(psi["value"]),
        "z": psi["z"]/psi["value"],
        "r": psi["r"]/psi["value"],
        "zz": (
            psi["zz"]/psi["value"]
            -(psi["z"]/psi["value"])**2
        ),
        "rr": (
            psi["rr"]/psi["value"]
            -(psi["r"]/psi["value"])**2
        ),
    }
    return {"psi": psi, "w": w, "Phi": reference["Phi"]}


def _radial_laplacian(field, r, *, regular_axis):
    result = np.empty_like(field["value"])
    positive = r > 0.0
    result[:, positive] = (
        field["rr"][:, positive]
        +2.0*field["r"][:, positive]/r[None, positive]
    )
    axis = ~positive
    if np.any(axis):
        if not regular_axis:
            raise ValueError(
                "analytic r=0 evaluation requires regular_axis=True"
            )
        result[:, axis] = 3.0*field["rr"][:, axis]
    return result


def _shape_laplacian(field, a, b, r, *, regular_axis):
    radial = _radial_laplacian(field, r, regular_axis=regular_axis)
    return (
        np.exp(-2.0*a["value"])*(
            field["zz"]-2.0*a["z"]*field["z"]
        )
        +np.exp(-2.0*b["value"])*(
            radial-2.0*b["r"]*field["r"]
        )
    )


def _shape_scalar_curvature(a, b, c, r, *, regular_axis):
    """Continuum scalar curvature of the unit-conformal shape metric."""
    aa = np.exp(a["value"])
    bb = np.exp(b["value"])
    cc = np.exp(c["value"])
    inverse_a2 = np.exp(-2.0*a["value"])
    inverse_b2 = np.exp(-2.0*b["value"])
    base = -2.0*(
        inverse_a2*(b["zz"]+b["z"]**2-a["z"]*b["z"])
        +inverse_b2*(a["rr"]+a["r"]**2-b["r"]*a["r"])
    )
    result = np.empty_like(base)
    positive = r > 0.0
    if np.any(positive):
        radius = r[None, positive]
        local_c = cc[:, positive]
        rho = radius*local_c
        rho_z = rho*c["z"][:, positive]
        rho_zz = rho*(
            c["zz"][:, positive]+c["z"][:, positive]**2
        )
        rho_r = local_c*(1.0+radius*c["r"][:, positive])
        rho_rr = local_c*(
            2.0*c["r"][:, positive]
            +radius*(c["rr"][:, positive]+c["r"][:, positive]**2)
        )
        ratio_a2_b2 = (aa[:, positive]/bb[:, positive])**2
        ratio_b2_a2 = 1.0/ratio_a2_b2
        hessian_zz = (
            rho_zz-a["z"][:, positive]*rho_z
            +ratio_a2_b2*a["r"][:, positive]*rho_r
        )
        hessian_rr = (
            rho_rr+ratio_b2_a2*b["z"][:, positive]*rho_z
            -b["r"][:, positive]*rho_r
        )
        laplacian_rho = (
            hessian_zz/aa[:, positive]**2
            +hessian_rr/bb[:, positive]**2
        )
        gradient_rho = (
            rho_z**2/aa[:, positive]**2
            +rho_r**2/bb[:, positive]**2
        )
        result[:, positive] = (
            base[:, positive]
            -4.0*laplacian_rho/rho
            +2.0*(1.0-gradient_rho)/rho**2
        )
    axis = ~positive
    if np.any(axis):
        if not regular_axis:
            raise ValueError(
                "analytic r=0 evaluation requires regular_axis=True"
            )
        hzz_over_rho = (
            c["zz"][:, axis]+c["z"][:, axis]**2
            -a["z"][:, axis]*c["z"][:, axis]
            +(aa[:, axis]/bb[:, axis])**2
            *(a["rr"][:, axis]+a["r"][:, axis]**2)
        )
        hrr_over_rho = (
            3.0*(c["rr"][:, axis]+c["r"][:, axis]**2)
            +(bb[:, axis]/aa[:, axis])**2
            *b["z"][:, axis]*c["z"][:, axis]
            -(b["rr"][:, axis]+b["r"][:, axis]**2)
        )
        ricci_zz = 0.5*base[:, axis]*aa[:, axis]**2-2.0*hzz_over_rho
        ricci_rr = 0.5*base[:, axis]*bb[:, axis]**2-2.0*hrr_over_rho
        result[:, axis] = (
            ricci_zz/aa[:, axis]**2
            +ricci_rr/bb[:, axis]**2
            +2.0*ricci_rr/cc[:, axis]**2
        )
    return result


def _analytic_raw_equations(primitives, chi, r, mass, *, regular_axis):
    psi = primitives["psi"]
    w = primitives["w"]
    phi = primitives["Phi"]
    a = primitives.get("a")
    b = primitives.get("b")
    c = primitives.get("c")
    if a is None:
        zero = {lane: np.zeros_like(psi["value"]) for lane in JET_LANES}
        a = b = c = zero
        scalar_bar = np.zeros_like(psi["value"])
    else:
        scalar_bar = _shape_scalar_curvature(
            a, b, c, r, regular_axis=regular_axis,
        )
    inverse_a = np.exp(-2.0*a["value"])
    inverse_b = np.exp(-2.0*b["value"])
    gradient = (
        inverse_a*(phi["z"]**2+chi["z"]**2)
        +inverse_b*(phi["r"]**2+chi["r"]**2)
    )
    laplacian_psi = _shape_laplacian(
        psi, a, b, r, regular_axis=regular_axis,
    )
    hamiltonian = (
        -6.0*laplacian_psi
        +(scalar_bar-gradient)*psi["value"]
        -(-12.0+mass*phi["value"]**2)*psi["value"]**3
    )
    laplacian_phi = _shape_laplacian(
        phi, a, b, r, regular_axis=regular_axis,
    )
    scalar = (
        laplacian_phi
        +3.0*(
            inverse_a*w["z"]*phi["z"]
            +inverse_b*w["r"]*phi["r"]
        )
        -mass*psi["value"]**2*phi["value"]
    )
    return {"hamiltonian": hamiltonian, "Phi": scalar}


def open_anisotropic_bulk_terms_from_jets(
    candidate_jets,
    reference_jets,
    z,
    r,
    background,
    *,
    regular_axis=False,
):
    """Evaluate open bulk terms from authoritative analytic spatial jets.

    Candidate fields are ``h_zz``, ``h_rr``, ``h_perp``, ``Phi``, and ``chi``.
    Each may be a nested mapping with ``value,z,r,zz,rr`` lanes or flat keys
    such as ``h_zz``, ``h_zz_z``, ..., ``h_zz_rr``.  Reference fields are
    ``q`` and ``Phi`` in the same form; the flat lowercase mapping returned by
    :meth:`FiniteWallReferenceJet.as_derivative_mapping` is accepted directly.

    Off-axis algebra is the default.  A mesh containing ``r=0`` is rejected
    unless ``regular_axis=True`` explicitly selects the regular even limits.
    No mixed derivative is used by these two equations.
    """
    z, r = _validated_coordinates(z, r, require_source_stencil=False)
    shape = (len(z), len(r))
    candidate = {
        "h_zz": _extract_jet(candidate_jets, ("h_zz",)),
        "h_rr": _extract_jet(candidate_jets, ("h_rr",)),
        "h_perp": _extract_jet(candidate_jets, ("h_perp",)),
        "Phi": _extract_jet(candidate_jets, ("Phi", "phi")),
        "chi": _extract_jet(candidate_jets, ("chi",)),
    }
    reference = {
        "q": _extract_jet(reference_jets, ("q",)),
        "Phi": _extract_jet(reference_jets, ("Phi", "phi")),
    }
    _validate_jet_shapes(candidate, shape)
    _validate_jet_shapes(reference, shape)
    if np.any(r == 0.0) and not bool(regular_axis):
        raise ValueError("analytic r=0 evaluation requires regular_axis=True")
    if np.any(r == 0.0):
        axis = np.flatnonzero(r == 0.0)
        if not np.array_equal(
            candidate["h_rr"]["value"][:, axis],
            candidate["h_perp"]["value"][:, axis],
        ):
            raise ValueError("regular-axis h_rr and h_perp values must agree exactly")

    primitive = _candidate_primitive_jets(candidate)
    reference_primitive = _reference_primitive_jets(reference, z)
    mass = float(background["mass_squared"])
    if not np.isfinite(mass):
        raise ValueError("bulk scalar mass must be finite")
    raw = _analytic_raw_equations(
        primitive,
        primitive["chi"],
        r,
        mass,
        regular_axis=bool(regular_axis),
    )
    defect = _analytic_raw_equations(
        reference_primitive,
        primitive["chi"],
        r,
        mass,
        regular_axis=bool(regular_axis),
    )
    return _result(
        raw,
        defect,
        method="analytic_candidate_and_reference_jets_open_bulk",
        axis_treatment=(
            "explicit_regular_even_limit" if np.any(r == 0.0)
            else "off_axis_only"
        ),
        source_stencil_width=None,
    )
