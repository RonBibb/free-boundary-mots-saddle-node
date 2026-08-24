"""Isolated primitives for the prospective matched staged-continuum audit.

Nothing in this module changes the V1 equations or production boundary
operators.  It builds one immutable continuous representation from an
overresolved parent, projects that representation directly to target grids,
constructs mode-neutral live-gauge cases, and exposes recoverable RK2 stage
records.  Full-matrix scheduling deliberately lives nowhere in this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np
from scipy.interpolate import BSpline, RectBivariateSpline, make_interp_spline

from bhps.anisotropic_geometry import anisotropic_hamiltonian_residual
from bhps.anisotropic_geometry import axisymmetric_diagonal_geometry
from bhps.anisotropic_initial_data import (
    _raw_residual_and_jacobian,
    anisotropic_initial_data_residual,
)
from bhps.gh_source_driver import (
    regular_so3_live_source_shift_advection,
    regular_so3_nonlinear_anchored_damped_wave_target,
    source_driver_rhs,
)
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.junction_preservation_diagnostic import (
    _orthonormal_frames,
    wall_junction_rows,
    wall_source_coefficients,
)
from bhps.junction_second_preservation_diagnostic import (
    wall_junction_second_tangent,
)
from bhps.nonlinear_regular_so3_evolution import (
    NativeRegularSO3RHS,
    StageRegularGaugeSource,
    apply_outer_source_sommerfeld,
    compact_wall_normal_gauge_position_residuals,
    compact_wall_position_residuals,
    gauge_constraint_summary,
    gauge_taylor_source_from_initial_jets,
    live_regular_source_second_time,
    regular_source_spatial_derivatives,
)
from bhps.regular_so3_gh_reduction import (
    FIELD_ORDER as REDUCED_FIELD_ORDER,
)
from bhps.staged_boundary_preservation import evaluate_boundary_stage_sequence


FIELD_COUNT = 9
PARENT_R_MAX = 12.0
TARGET_R_MAX = 10.0
STENCIL_WIDTH = 7
BOUNDARY_MODES = (
    "legacy_wall_axis_outer",
    "wall_owner_last_experimental",
)
MANDATORY_LANDMARKS = {
    "legacy_wall_axis_outer": (
        "bulk_positive_radius",
        "initial_axis_fill",
        "final_compact_wall_endpoint_solve",
        "final_compact_post_wall_axis_fill",
        "pre_outer",
        "post_outer",
        "post_axis_operator_repair",
    ),
    "wall_owner_last_experimental": (
        "bulk_positive_radius",
        "initial_axis_fill",
        "outer_open_face_before_wall",
        "coupled_Phi_gzz_wall_solve",
        "final_compact_wall_endpoint_solve",
        "post_wall_owner_reconciliation",
        "post_axis_operator_repair",
    ),
}


def _immutable_array(value, dtype=None):
    """Return a C-contiguous array backed by immutable bytes."""
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _jsonable(value):
    """Convert nested scientific metadata to a stable JSON value."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    raise TypeError(f"metadata value {type(value).__name__} is not JSON-safe")


def _freeze_value(value):
    """Deep-freeze mappings/sequences and copy arrays onto immutable buffers."""
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _freeze_value(item) for key, item in value.items()
        })
    if isinstance(value, np.ndarray):
        return _immutable_array(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _update_digest(digest, name, value):
    """Hash a named nested value without relying on object identity."""
    digest.update(str(name).encode())
    digest.update(b"\0")
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value):
            _update_digest(digest, key, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for index, item in enumerate(value):
            _update_digest(digest, index, item)
        return
    if isinstance(value, (str, bool, int, float)) or value is None:
        digest.update(json.dumps(value, sort_keys=True).encode())
        return
    array = np.ascontiguousarray(value)
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())


@dataclass(frozen=True)
class TargetGridSpec:
    label: str
    nz: int
    nr_r12: int
    nr_r10: int


TARGET_GRIDS = {
    "G8": TargetGridSpec("G8", 97, 217, 181),
    "G9": TargetGridSpec("G9", 113, 253, 211),
    "G10": TargetGridSpec("G10", 129, 289, 241),
}


@dataclass(frozen=True)
class DriverConfiguration:
    driver_mu: float = 2.0
    driver_eta: float = 1.25
    target_mu_lapse: float = 0.4
    target_mu_shift: float = 0.6
    target_power: float = 0.5
    stencil_width: int = STENCIL_WIDTH
    radial_constraint_cut: float = 6.0

    def __post_init__(self):
        frozen = (2.0, 1.25, 0.4, 0.6, 0.5, STENCIL_WIDTH, 6.0)
        found = (
            float(self.driver_mu), float(self.driver_eta),
            float(self.target_mu_lapse), float(self.target_mu_shift),
            float(self.target_power), int(self.stencil_width),
            float(self.radial_constraint_cut),
        )
        if found != frozen:
            raise ValueError("matched continuum driver configuration is frozen")

    def public(self):
        return {
            "driver_mu": self.driver_mu,
            "driver_eta": self.driver_eta,
            "target_mu_lapse": self.target_mu_lapse,
            "target_mu_shift": self.target_mu_shift,
            "target_power": self.target_power,
            "stencil_width": self.stencil_width,
            "radial_constraint_cut": self.radial_constraint_cut,
        }


@dataclass(frozen=True)
class ProjectedJetField:
    z: np.ndarray
    r: np.ndarray
    reduced_fields: np.ndarray
    reduced_first: np.ndarray
    reduced_second: np.ndarray
    primitive_fields: dict | None = None

    def __post_init__(self):
        z = _immutable_array(self.z, float)
        r = _immutable_array(self.r, float)
        q = _immutable_array(self.reduced_fields, float)
        first = _immutable_array(self.reduced_first, float)
        second = _immutable_array(self.reduced_second, float)
        expected = (len(z), len(r), FIELD_COUNT)
        if q.shape != expected or first.shape != (3, *expected):
            raise ValueError("invalid projected fields or first jets")
        if second.shape != (3, 3, *expected):
            raise ValueError("invalid projected second jets")
        if not all(np.all(np.isfinite(value)) for value in (z, r, q, first, second)):
            raise ValueError("projected jets must be finite")
        primitives = None
        if self.primitive_fields is not None:
            frozen = {}
            for name, value in self.primitive_fields.items():
                array = _immutable_array(value, float)
                if array.shape != expected[:2] or not np.all(np.isfinite(array)):
                    raise ValueError(f"invalid projected primitive {name}")
                frozen[str(name)] = array
            primitives = MappingProxyType(frozen)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "reduced_fields", q)
        object.__setattr__(self, "reduced_first", first)
        object.__setattr__(self, "reduced_second", second)
        object.__setattr__(self, "primitive_fields", primitives)


@dataclass(frozen=True)
class TensorSplineSurface:
    """Tensor B-spline coefficients for fields on ``(z,s=(r/R)^2)``."""

    z_knots: np.ndarray
    s_knots: np.ndarray
    coefficients: np.ndarray
    z_degree: int
    s_degree: int
    parent_r_max: float
    z_boundary: str

    def __post_init__(self):
        z_knots = _immutable_array(self.z_knots, float)
        s_knots = _immutable_array(self.s_knots, float)
        coefficients = _immutable_array(self.coefficients, float)
        z_degree = int(self.z_degree)
        s_degree = int(self.s_degree)
        parent_r_max = float(self.parent_r_max)
        if z_knots.ndim != 1 or s_knots.ndim != 1 or coefficients.ndim < 2:
            raise ValueError("invalid tensor-spline coefficient arrays")
        if z_degree < 1 or s_degree < 1 or parent_r_max <= 0.0:
            raise ValueError("invalid tensor-spline metadata")
        if coefficients.shape[0] != len(z_knots)-z_degree-1:
            raise ValueError("z coefficient/knots mismatch")
        if coefficients.shape[1] != len(s_knots)-s_degree-1:
            raise ValueError("s coefficient/knots mismatch")
        if not all(np.all(np.isfinite(value)) for value in (
            z_knots, s_knots, coefficients,
        )):
            raise ValueError("tensor-spline arrays must be finite")
        object.__setattr__(self, "z_knots", z_knots)
        object.__setattr__(self, "s_knots", s_knots)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "z_degree", z_degree)
        object.__setattr__(self, "s_degree", s_degree)
        object.__setattr__(self, "parent_r_max", parent_r_max)
        object.__setattr__(self, "z_boundary", str(self.z_boundary))

    @classmethod
    def build(
        cls, z, r, values, *, z_first=None, degree=3,
        parent_r_max=PARENT_R_MAX,
    ):
        z = np.asarray(z, dtype=float)
        r = np.asarray(r, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.shape[:2] != (len(z), len(r)):
            raise ValueError("surface values are not aligned with z,r")
        if int(degree) < 1 or len(z) <= degree or len(r) <= degree:
            raise ValueError("insufficient nodes for requested spline degree")
        if np.any(np.diff(z) <= 0.0) or not np.all(np.isfinite(z)):
            raise ValueError("compact nodes must be finite and increase")
        if r[0] != 0.0 or np.any(np.diff(r) <= 0.0):
            raise ValueError("radial nodes must start at zero and increase")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(values)):
            raise ValueError("surface nodes and values must be finite")
        scale = float(parent_r_max)
        if scale <= 0.0 or r[-1] > scale + 1e-12:
            raise ValueError("invalid parent radial scale")
        s = (r / scale) ** 2
        radial = make_interp_spline(s, values, k=int(degree), axis=1)
        boundary = "not-a-knot"
        bc_type = None
        if z_first is not None:
            if int(degree) != 3:
                raise ValueError("first-derivative clamping is primary cubic only")
            z_first = np.asarray(z_first, dtype=float)
            if z_first.shape != values.shape:
                raise ValueError("z boundary derivative shape mismatch")
            lower = make_interp_spline(
                s, z_first[0], k=int(degree), axis=0,
            )
            upper = make_interp_spline(
                s, z_first[-1], k=int(degree), axis=0,
            )
            if not (
                np.array_equal(lower.t, radial.t)
                and np.array_equal(upper.t, radial.t)
            ):
                raise RuntimeError("radial spline bases are not identical")
            bc_type = ([(1, lower.c)], [(1, upper.c)])
            boundary = "clamped_parent_endpoint_z_first"
        # SciPy stores the interpolation coefficient axis first even when the
        # input interpolation axis is one.  Restore (z, s-coefficient, field)
        # before constructing the z spline.
        radial_coefficients = np.moveaxis(radial.c, 0, 1)
        compact = make_interp_spline(
            z, radial_coefficients, k=int(degree), axis=0,
            bc_type=bc_type,
        )
        return cls(
            np.asarray(compact.t).copy(), np.asarray(radial.t).copy(),
            np.asarray(compact.c).copy(), int(degree), int(degree), scale,
            boundary,
        )

    def evaluate(self, z, r, z_order=0, s_order=0):
        z = np.asarray(z, dtype=float)
        r = np.asarray(r, dtype=float)
        z_min = self.z_knots[self.z_degree]
        z_max = self.z_knots[-self.z_degree-1]
        if (
            np.any(~np.isfinite(z)) or np.any(~np.isfinite(r))
            or np.any(z < z_min) or np.any(z > z_max)
            or np.any(r < 0.0) or np.any(r > self.parent_r_max)
        ):
            raise ValueError("tensor-spline evaluation lies outside its domain")
        s = (r / self.parent_r_max) ** 2
        radial = BSpline(
            self.s_knots, self.coefficients, self.s_degree, axis=1,
            extrapolate=False,
        )(s, nu=int(s_order))
        return BSpline(
            self.z_knots, radial, self.z_degree, axis=0,
            extrapolate=False,
        )(z, nu=int(z_order))

    def coefficient_arrays(self, prefix):
        return {
            f"{prefix}_z_knots": self.z_knots.copy(),
            f"{prefix}_s_knots": self.s_knots.copy(),
            f"{prefix}_coefficients": self.coefficients.copy(),
            f"{prefix}_z_degree": np.asarray(self.z_degree),
            f"{prefix}_s_degree": np.asarray(self.s_degree),
            f"{prefix}_parent_r_max": np.asarray(self.parent_r_max),
            f"{prefix}_z_boundary": np.asarray(self.z_boundary),
        }

    @classmethod
    def from_arrays(cls, archive, prefix):
        return cls(
            np.asarray(archive[f"{prefix}_z_knots"]),
            np.asarray(archive[f"{prefix}_s_knots"]),
            np.asarray(archive[f"{prefix}_coefficients"]),
            int(archive[f"{prefix}_z_degree"]),
            int(archive[f"{prefix}_s_degree"]),
            float(archive[f"{prefix}_parent_r_max"]),
            str(archive[f"{prefix}_z_boundary"]),
        )


@dataclass(frozen=True)
class ContinuousReducedParent:
    position: TensorSplineSurface
    velocity: TensorSplineSurface
    acceleration: TensorSplineSurface
    parent_identity: str = "unspecified"
    parent_nodal_fingerprint: str = ""
    field_order: tuple = tuple(REDUCED_FIELD_ORDER)

    def __post_init__(self):
        surfaces = (self.position, self.velocity, self.acceleration)
        scales = {surface.parent_r_max for surface in surfaces}
        domains = {
            (
                surface.z_knots[surface.z_degree],
                surface.z_knots[-surface.z_degree-1],
            ) for surface in surfaces
        }
        radial_domains = {
            (
                surface.s_knots[surface.s_degree],
                surface.s_knots[-surface.s_degree-1],
            ) for surface in surfaces
        }
        if len(scales) != 1 or len(domains) != 1 or len(radial_domains) != 1:
            raise ValueError("continuous-parent surfaces have inconsistent domains")
        if any(surface.coefficients.shape[-1] != FIELD_COUNT for surface in surfaces):
            raise ValueError("continuous-parent surface field count mismatch")
        order = tuple(str(value) for value in self.field_order)
        if order != tuple(REDUCED_FIELD_ORDER):
            raise ValueError("continuous-parent reduced field order mismatch")
        object.__setattr__(self, "parent_identity", str(self.parent_identity))
        object.__setattr__(self, "parent_nodal_fingerprint", str(
            self.parent_nodal_fingerprint
        ))
        object.__setattr__(self, "field_order", order)

    @classmethod
    def from_jet_field(
        cls, jet_field, z, r, *, degree=3, parent_r_max=PARENT_R_MAX,
        parent_identity="unspecified", expected_shape=None,
        require_full_radial_domain=False,
    ):
        z = np.asarray(z, dtype=float)
        r = np.asarray(r, dtype=float)
        if not hasattr(jet_field, "z") or not hasattr(jet_field, "r"):
            raise ValueError("jet field must expose its source z,r coordinates")
        if not (
            np.array_equal(np.asarray(jet_field.z, dtype=float), z)
            and np.array_equal(np.asarray(jet_field.r, dtype=float), r)
        ):
            raise ValueError("jet field coordinates do not match supplied z,r")
        if expected_shape is not None and (len(z), len(r)) != tuple(expected_shape):
            raise ValueError("continuous-parent source grid does not match its identity")
        if require_full_radial_domain and not np.isclose(
            r[-1], float(parent_r_max), rtol=0.0, atol=1e-12,
        ):
            raise ValueError("canonical parent must cover the full radial domain")
        q = np.asarray(jet_field.reduced_fields, dtype=float)
        first = np.asarray(jet_field.reduced_first, dtype=float)
        second = np.asarray(jet_field.reduced_second, dtype=float)
        expected = (len(z), len(r), FIELD_COUNT)
        if q.shape != expected or first.shape != (3, *expected):
            raise ValueError("invalid parent reduced fields/first jets")
        if second.shape != (3, 3, *expected):
            raise ValueError("invalid parent reduced second jets")
        if not all(np.all(np.isfinite(value)) for value in (
            z, r, q, first, second,
        )):
            raise ValueError("continuous-parent nodal data must be finite")
        nodal_fingerprint = hash_arrays(z, r, q, first, second)
        # Position and velocity have available endpoint z derivatives.  The
        # acceleration has no archived q_ttz jet, so it is not-a-knot in z.
        return cls(
            TensorSplineSurface.build(
                z, r, q, z_first=first[1], degree=degree,
                parent_r_max=parent_r_max,
            ),
            TensorSplineSurface.build(
                z, r, first[0], z_first=second[0, 1], degree=degree,
                parent_r_max=parent_r_max,
            ),
            TensorSplineSurface.build(
                z, r, second[0, 0], degree=degree,
                parent_r_max=parent_r_max,
            ),
            str(parent_identity), nodal_fingerprint,
            tuple(REDUCED_FIELD_ORDER),
        )

    def project(self, z, r):
        z = np.asarray(z, dtype=float)
        r = np.asarray(r, dtype=float)
        q = self.position.evaluate(z, r)
        v = self.velocity.evaluate(z, r)
        a = self.acceleration.evaluate(z, r)
        q_z = self.position.evaluate(z, r, z_order=1)
        q_zz = self.position.evaluate(z, r, z_order=2)
        q_s = self.position.evaluate(z, r, s_order=1)
        q_ss = self.position.evaluate(z, r, s_order=2)
        q_zs = self.position.evaluate(z, r, z_order=1, s_order=1)
        v_z = self.velocity.evaluate(z, r, z_order=1)
        v_s = self.velocity.evaluate(z, r, s_order=1)
        ds_dr = 2.0 * r / self.position.parent_r_max**2
        d2s_dr2 = 2.0 / self.position.parent_r_max**2
        first = np.zeros((3, len(z), len(r), FIELD_COUNT))
        second = np.zeros((3, 3, len(z), len(r), FIELD_COUNT))
        first[0] = v
        first[1] = q_z
        first[2] = q_s * ds_dr[None, :, None]
        second[0, 0] = a
        second[0, 1] = second[1, 0] = v_z
        second[0, 2] = second[2, 0] = v_s * ds_dr[None, :, None]
        second[1, 1] = q_zz
        second[1, 2] = second[2, 1] = (
            q_zs * ds_dr[None, :, None]
        )
        second[2, 2] = (
            q_ss * ds_dr[None, :, None] ** 2
            + q_s * d2s_dr2
        )
        if not all(np.all(np.isfinite(value)) for value in (q, first, second)):
            raise RuntimeError("nonfinite continuous-parent projection")
        return ProjectedJetField(z.copy(), r.copy(), q, first, second)

    def coefficient_arrays(self):
        return {
            **self.position.coefficient_arrays("position"),
            **self.velocity.coefficient_arrays("velocity"),
            **self.acceleration.coefficient_arrays("acceleration"),
            "parent_identity": np.asarray(self.parent_identity),
            "parent_nodal_fingerprint": np.asarray(
                self.parent_nodal_fingerprint
            ),
            "reduced_field_order": np.asarray(self.field_order),
        }

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.coefficient_arrays().items()):
            _update_digest(digest, name, value)
        return digest.hexdigest()

    @classmethod
    def from_arrays(cls, archive):
        return cls(
            *(TensorSplineSurface.from_arrays(archive, prefix)
              for prefix in ("position", "velocity", "acceleration")),
            parent_identity=str(archive["parent_identity"]),
            parent_nodal_fingerprint=str(archive["parent_nodal_fingerprint"]),
            field_order=tuple(str(value) for value in archive["reduced_field_order"]),
        )


@dataclass(frozen=True)
class ContinuousPrimitiveParent:
    """Primitive P11 fields needed for independent constraint reconstruction."""

    surfaces: dict
    parent_identity: str = "unspecified"
    source_coordinate_fingerprint: str = ""
    primitive_nodal_fingerprint: str = ""

    def __post_init__(self):
        surfaces = {str(name): value for name, value in self.surfaces.items()}
        required = {"psi", "a", "b", "c", "phi", "chi"}
        if set(surfaces) != required:
            raise ValueError("primitive parent has an incomplete field set")
        scales = {surface.parent_r_max for surface in surfaces.values()}
        domains = {
            (
                surface.z_knots[surface.z_degree],
                surface.z_knots[-surface.z_degree-1],
                surface.s_knots[surface.s_degree],
                surface.s_knots[-surface.s_degree-1],
            ) for surface in surfaces.values()
        }
        if len(scales) != 1 or len(domains) != 1:
            raise ValueError("primitive-parent surfaces have inconsistent domains")
        object.__setattr__(self, "surfaces", MappingProxyType(surfaces))
        object.__setattr__(self, "parent_identity", str(self.parent_identity))
        object.__setattr__(self, "source_coordinate_fingerprint", str(
            self.source_coordinate_fingerprint
        ))
        object.__setattr__(self, "primitive_nodal_fingerprint", str(
            self.primitive_nodal_fingerprint
        ))

    @classmethod
    def from_geometry(
        cls, geometry, degree=3, *, expected_shape=None,
        require_full_radial_domain=False,
    ):
        z = np.asarray(geometry["z"], dtype=float)
        r = np.asarray(geometry["r"], dtype=float)
        if expected_shape is not None and (len(z), len(r)) != tuple(expected_shape):
            raise ValueError("primitive-parent source grid has the wrong shape")
        if require_full_radial_domain and not np.isclose(
            r[-1], PARENT_R_MAX, rtol=0.0, atol=1e-12,
        ):
            raise ValueError("primitive parent does not cover R12")
        jet = geometry["jet_field"]
        if not (
            np.array_equal(np.asarray(jet.z, dtype=float), z)
            and np.array_equal(np.asarray(jet.r, dtype=float), r)
        ):
            raise ValueError("primitive parent and jet coordinates differ")
        primitive = {
            "psi": np.asarray(geometry["psi"], dtype=float),
            "a": np.asarray(geometry["a"], dtype=float),
            "b": np.asarray(geometry["b"], dtype=float),
            "c": np.asarray(geometry["c"], dtype=float),
            "phi": np.asarray(geometry["phi"], dtype=float),
            "chi": np.asarray(jet.reduced_fields[:, :, 8], dtype=float),
        }
        if not all(value.shape == (len(z), len(r)) for value in primitive.values()):
            raise ValueError("primitive parent fields have the wrong shape")
        names = sorted(primitive)
        return cls({
            name: TensorSplineSurface.build(
                z, r, value[:, :, None],
                degree=degree, parent_r_max=PARENT_R_MAX,
            )
            for name, value in primitive.items()
        }, str(geometry["name"]), hash_arrays(z, r), hash_arrays(
            z, r, *(primitive[name] for name in names),
        ))

    def project(self, z, r):
        return {
            name: surface.evaluate(z, r)[:, :, 0]
            for name, surface in self.surfaces.items()
        }

    def coefficient_arrays(self):
        output = {
            "primitive_names": np.asarray(sorted(self.surfaces)),
            "primitive_parent_identity": np.asarray(self.parent_identity),
            "primitive_source_coordinate_fingerprint": np.asarray(
                self.source_coordinate_fingerprint
            ),
            "primitive_nodal_fingerprint": np.asarray(
                self.primitive_nodal_fingerprint
            ),
        }
        for name in sorted(self.surfaces):
            output.update(self.surfaces[name].coefficient_arrays(
                f"primitive_{name}"
            ))
        return output

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.coefficient_arrays().items()):
            _update_digest(digest, name, value)
        return digest.hexdigest()

    @classmethod
    def from_arrays(cls, archive):
        names = [str(value) for value in archive["primitive_names"]]
        return cls({
            name: TensorSplineSurface.from_arrays(archive, f"primitive_{name}")
            for name in names
        }, str(archive["primitive_parent_identity"]),
            str(archive["primitive_source_coordinate_fingerprint"]),
            str(archive["primitive_nodal_fingerprint"]))


def target_coordinates(parent_z, specification, r_max=TARGET_R_MAX):
    if not isinstance(specification, TargetGridSpec):
        raise TypeError("target specification is required")
    if TARGET_GRIDS.get(specification.label) != specification:
        raise ValueError("only a frozen G8/G9/G10 target specification is valid")
    if not np.isclose(float(r_max), TARGET_R_MAX, rtol=0.0, atol=0.0):
        raise ValueError("matched continuum targets are frozen at R10")
    parent_z = np.asarray(parent_z, dtype=float)
    if (
        parent_z.ndim != 1 or np.any(np.diff(parent_z) <= 0.0)
        or not np.all(np.isfinite(parent_z))
        or not np.isclose(parent_z[0], 1.0, rtol=0.0, atol=1e-14)
        or not np.isclose(parent_z[-1], np.e, rtol=0.0, atol=1e-14)
    ):
        raise ValueError("parent compact interval must be the canonical [1,e]")
    parent_dr = PARENT_R_MAX/(specification.nr_r12-1)
    target_dr = TARGET_R_MAX/(specification.nr_r10-1)
    if not np.isclose(parent_dr, target_dr, rtol=0.0, atol=1e-14):
        raise RuntimeError("frozen R12 and R10 radial spacings do not match")
    z = _immutable_array(np.linspace(
        parent_z[0], parent_z[-1], specification.nz,
    ), float)
    r = _immutable_array(np.linspace(
        0.0, float(r_max), specification.nr_r10,
    ), float)
    return z, r


def normalized_error(reference, comparison):
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    if reference.shape != comparison.shape:
        raise ValueError("error arrays must have equal shapes")
    difference = comparison - reference
    scale = max(
        1.0, float(np.max(np.abs(reference))),
        float(np.max(np.abs(comparison))),
    )
    return {
        "scaled_Linf": float(np.max(np.abs(difference)) / scale),
        "relative_L2": float(np.linalg.norm(difference) / max(
            np.linalg.norm(reference), np.linalg.norm(comparison), 1e-300,
        )),
        "maximum_absolute": float(np.max(np.abs(difference))),
        "finite": bool(np.all(np.isfinite(difference))),
    }


def hash_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def projection_fingerprint(jet_field):
    digest = hashlib.sha256()
    for name, value in (
        ("z", jet_field.z), ("r", jet_field.r),
        ("reduced_fields", jet_field.reduced_fields),
        ("reduced_first", jet_field.reduced_first),
        ("reduced_second", jet_field.reduced_second),
    ):
        _update_digest(digest, name, value)
    for name, value in sorted((jet_field.primitive_fields or {}).items()):
        _update_digest(digest, f"primitive_{name}", value)
    _update_digest(digest, "field_order", tuple(REDUCED_FIELD_ORDER))
    return digest.hexdigest()


def _validate_primary_recipe(representation):
    surfaces = {
        "position": representation.position,
        "velocity": representation.velocity,
        "acceleration": representation.acceleration,
    }
    for name, surface in surfaces.items():
        if (
            surface.parent_r_max != PARENT_R_MAX
            or surface.z_degree != 3 or surface.s_degree != 3
            or not np.isclose(
                surface.z_knots[surface.z_degree], 1.0,
                rtol=0.0, atol=1e-14,
            )
            or not np.isclose(
                surface.z_knots[-surface.z_degree-1], np.e,
                rtol=0.0, atol=1e-14,
            )
            or not np.isclose(
                surface.s_knots[surface.s_degree], 0.0,
                rtol=0.0, atol=1e-15,
            )
            or not np.isclose(
                surface.s_knots[-surface.s_degree-1], 1.0,
                rtol=0.0, atol=1e-15,
            )
        ):
            raise ValueError(f"{name} surface violates the frozen cubic recipe")
    if (
        surfaces["position"].z_boundary
        != "clamped_parent_endpoint_z_first"
        or surfaces["velocity"].z_boundary
        != "clamped_parent_endpoint_z_first"
        or surfaces["acceleration"].z_boundary != "not-a-knot"
    ):
        raise ValueError("continuous-parent endpoint rules are not frozen")


def projected_geometry(
    parent_geometry, representation, specification,
    primitive_representation=None,
):
    parent_z = np.asarray(parent_geometry["z"], dtype=float)
    parent_r = np.asarray(parent_geometry["r"], dtype=float)
    canonical_z = np.linspace(1.0, np.e, 145)
    canonical_r = np.linspace(0.0, PARENT_R_MAX, 325)
    if (len(parent_z), len(parent_r)) != (145, 325):
        raise ValueError("production projections require the canonical P11 grid")
    if not (
        np.array_equal(parent_z, canonical_z)
        and np.array_equal(parent_r, canonical_r)
        and np.array_equal(parent_z, np.asarray(parent_geometry["jet_field"].z))
        and np.array_equal(parent_r, np.asarray(parent_geometry["jet_field"].r))
        and np.isclose(parent_r[0], 0.0)
        and np.isclose(parent_r[-1], PARENT_R_MAX)
    ):
        raise ValueError("P11 parent coordinates are inconsistent")
    if not np.isclose(
        float(parent_geometry.get("fold_amplitude", np.nan)), 7.90,
        rtol=0.0, atol=1e-14,
    ):
        raise ValueError("production parent is not the frozen A=7.90 state")
    _validate_primary_recipe(representation)
    if representation.parent_identity != str(parent_geometry["name"]):
        raise ValueError("continuous representation parent identity mismatch")
    if representation.parent_nodal_fingerprint != hash_arrays(
        parent_z, parent_r,
        parent_geometry["jet_field"].reduced_fields,
        parent_geometry["jet_field"].reduced_first,
        parent_geometry["jet_field"].reduced_second,
    ):
        raise ValueError("continuous representation nodal fingerprint mismatch")
    if primitive_representation is None:
        raise ValueError("production projection requires the bound primitive parent")
    primitive_values = {
        "psi": np.asarray(parent_geometry["psi"]),
        "a": np.asarray(parent_geometry["a"]),
        "b": np.asarray(parent_geometry["b"]),
        "c": np.asarray(parent_geometry["c"]),
        "phi": np.asarray(parent_geometry["phi"]),
        "chi": np.asarray(parent_geometry["jet_field"].reduced_fields[:, :, 8]),
    }
    if (
        primitive_representation.parent_identity != str(parent_geometry["name"])
        or primitive_representation.source_coordinate_fingerprint
        != hash_arrays(parent_z, parent_r)
        or primitive_representation.primitive_nodal_fingerprint
        != hash_arrays(
            parent_z, parent_r,
            *(primitive_values[name] for name in sorted(primitive_values)),
        )
    ):
        raise ValueError("primitive representation is not bound to this P11")
    for name, surface in primitive_representation.surfaces.items():
        if (
            surface.parent_r_max != PARENT_R_MAX
            or surface.z_degree != 3 or surface.s_degree != 3
            or surface.z_boundary != "not-a-knot"
            or not np.isclose(surface.z_knots[3], 1.0, rtol=0.0, atol=1e-14)
            or not np.isclose(surface.z_knots[-4], np.e, rtol=0.0, atol=1e-14)
            or not np.isclose(surface.s_knots[3], 0.0, rtol=0.0, atol=1e-15)
            or not np.isclose(surface.s_knots[-4], 1.0, rtol=0.0, atol=1e-15)
        ):
            raise ValueError(f"primitive surface {name} violates the frozen recipe")
    z, r = target_coordinates(parent_geometry["z"], specification)
    base_jet = representation.project(z, r)
    primitive_fields = (
        None if primitive_representation is None
        else primitive_representation.project(z, r)
    )
    jet = ProjectedJetField(
        base_jet.z, base_jet.r, base_jet.reduced_fields,
        base_jet.reduced_first, base_jet.reduced_second,
        primitive_fields,
    )
    return {
        "name": f"{specification.label}-A790-P11-direct-R10-projection",
        "source_grid": [len(parent_z), len(parent_r)],
        "target_grid": [len(z), len(r)],
        "z": jet.z, "r": jet.r,
        "background": _freeze_value(parent_geometry["background"]),
        "mass_squared": float(parent_geometry["mass_squared"]),
        "fold_amplitude": float(parent_geometry["fold_amplitude"]),
        "radial_domain": [0.0, TARGET_R_MAX],
        "jet_field": jet,
        "continuous_parent": str(parent_geometry["name"]),
        "continuous_parent_fingerprint": representation.fingerprint(),
        "primitive_parent_fingerprint": (
            None if primitive_representation is None
            else primitive_representation.fingerprint()
        ),
        "projection_fingerprint": projection_fingerprint(jet),
        "projection": {
            "grid_label": str(specification.label),
            "coordinates": "z,s=(r/12)^2",
            "primary": "tensor_product_cubic_clamped_by_parent_endpoint_q_z",
            "target_to_target_interpolation": False,
            "endpoint_repair": False,
            "constraint_resolve": False,
            "field_order": list(REDUCED_FIELD_ORDER),
        },
    }


def quintic_adverse_projection(jet_field, z_parent, r_parent, z_target, r_target):
    """Zero-smoothing quintic RectBivariateSpline adverse comparator."""
    z_parent = np.asarray(z_parent, dtype=float)
    r_parent = np.asarray(r_parent, dtype=float)
    z_target = np.asarray(z_target, dtype=float)
    r_target = np.asarray(r_target, dtype=float)
    if not (
        np.array_equal(z_parent, np.asarray(jet_field.z, dtype=float))
        and np.array_equal(r_parent, np.asarray(jet_field.r, dtype=float))
    ):
        raise ValueError("adverse comparator coordinates differ from source jet")
    s_parent = (r_parent / PARENT_R_MAX) ** 2
    s_target = (r_target / PARENT_R_MAX) ** 2
    zz, ss = np.meshgrid(z_target, s_target, indexing="ij")

    def evaluate(values, dx=0, dy=0):
        output = np.empty((len(z_target), len(r_target), FIELD_COUNT))
        for field in range(FIELD_COUNT):
            spline = RectBivariateSpline(
                z_parent, s_parent, np.asarray(values)[:, :, field],
                kx=5, ky=5, s=0,
            )
            output[:, :, field] = spline.ev(
                zz.ravel(), ss.ravel(), dx=dx, dy=dy,
            ).reshape(len(z_target), len(r_target))
        return output

    q = evaluate(jet_field.reduced_fields)
    v = evaluate(jet_field.reduced_first[0])
    a = evaluate(jet_field.reduced_second[0, 0])
    qz = evaluate(jet_field.reduced_fields, dx=1)
    qs = evaluate(jet_field.reduced_fields, dy=1)
    qzz = evaluate(jet_field.reduced_fields, dx=2)
    qzs = evaluate(jet_field.reduced_fields, dx=1, dy=1)
    qss = evaluate(jet_field.reduced_fields, dy=2)
    vz = evaluate(jet_field.reduced_first[0], dx=1)
    vs = evaluate(jet_field.reduced_first[0], dy=1)
    ds = 2.0 * r_target / PARENT_R_MAX**2
    d2s = 2.0 / PARENT_R_MAX**2
    first = np.zeros((3, len(z_target), len(r_target), FIELD_COUNT))
    second = np.zeros((3, 3, len(z_target), len(r_target), FIELD_COUNT))
    first[0] = v; first[1] = qz; first[2] = qs * ds[None, :, None]
    second[0, 0] = a; second[0, 1] = second[1, 0] = vz
    second[0, 2] = second[2, 0] = vs * ds[None, :, None]
    second[1, 1] = qzz
    second[1, 2] = second[2, 1] = qzs * ds[None, :, None]
    second[2, 2] = qss * ds[None, :, None]**2 + qs * d2s
    return ProjectedJetField(
        z_target.copy(), r_target.copy(), q, first, second,
    )


def representation_reconstruction(parent_jet, representation, z, r):
    """Score parent-node reconstruction and stored-jet comparator lanes."""
    projected = representation.project(z, r)
    return {
        "position": normalized_error(
            parent_jet.reduced_fields, projected.reduced_fields,
        ),
        "velocity": normalized_error(
            parent_jet.reduced_first[0], projected.reduced_first[0],
        ),
        "acceleration": normalized_error(
            parent_jet.reduced_second[0, 0], projected.reduced_second[0, 0],
        ),
        "first_spatial_stored_comparator": normalized_error(
            parent_jet.reduced_first[1:], projected.reduced_first[1:],
        ),
        "second_spatial_stored_comparator": normalized_error(
            parent_jet.reduced_second[1:, 1:],
            projected.reduced_second[1:, 1:],
        ),
        "endpoint_z_first_stored_comparator": normalized_error(
            parent_jet.reduced_first[1, [0, -1]],
            projected.reduced_first[1, [0, -1]],
        ),
        "endpoint_velocity_z_first_stored_comparator": normalized_error(
            parent_jet.reduced_second[0, 1, [0, -1]],
            projected.reduced_second[0, 1, [0, -1]],
        ),
    }


def round_trip_to_parent(projected, parent_on_r10):
    """Interpolate a target back to P11 nodes as a diagnostic only."""
    coarse_parent = ContinuousReducedParent.from_jet_field(
        projected, projected.z, projected.r, degree=3,
        parent_r_max=PARENT_R_MAX,
    )
    returned = coarse_parent.project(parent_on_r10.z, parent_on_r10.r)
    return {
        "position": normalized_error(
            parent_on_r10.reduced_fields, returned.reduced_fields,
        ),
        "velocity": normalized_error(
            parent_on_r10.reduced_first[0], returned.reduced_first[0],
        ),
        "acceleration": normalized_error(
            parent_on_r10.reduced_second[0, 0], returned.reduced_second[0, 0],
        ),
        "first_spatial": normalized_error(
            parent_on_r10.reduced_first[1:], returned.reduced_first[1:],
        ),
        "second_spatial": normalized_error(
            parent_on_r10.reduced_second[1:, 1:],
            returned.reduced_second[1:, 1:],
        ),
    }


def reconstruct_anisotropic_fields(projected):
    """Reconstruct the physical ansatz exactly from q2/q3/q4/q6.

    The time-symmetric parent has ``alpha=psi``.  Independently projected
    primitive fields are comparator lanes only and never define the state.
    """
    q = np.asarray(projected.reduced_fields, dtype=float)
    r = np.asarray(projected.r, dtype=float)
    if q.shape != (len(projected.z), len(r), FIELD_COUNT):
        raise ValueError("invalid projected reduced field shape")
    lapse_squared = -q[:, :, 2]
    transverse = q[:, :, 3]
    radial = transverse + r[None, :] ** 2 * q[:, :, 4]
    compact = q[:, :, 6]
    if np.any(lapse_squared <= 0.0) or np.any(
        np.stack((transverse, radial, compact)) <= 0.0
    ):
        raise ValueError("projected metric cannot be inverted anisotropically")
    psi = np.sqrt(lapse_squared)
    a = 0.5 * np.log(compact / psi**2)
    b = 0.5 * np.log(radial / psi**2)
    c = 0.5 * np.log(transverse / psi**2)
    return {
        "alpha": psi,
        "psi": psi, "a": a, "b": b, "c": c,
        "phi": q[:, :, 7], "chi": q[:, :, 8],
        "chi_z": projected.reduced_first[1, :, :, 8],
        "chi_r": projected.reduced_first[2, :, :, 8],
        "tracefree_maximum_absolute": float(np.max(np.abs(a + b + 2.0*c))),
        "primitive_comparator": (
            {} if projected.primitive_fields is None else {
                name: normalized_error(
                    np.asarray(projected.primitive_fields[name]), value,
                )
                for name, value in (
                    ("psi", psi), ("a", a), ("b", b), ("c", c),
                    ("phi", q[:, :, 7]), ("chi", q[:, :, 8]),
                ) if name in projected.primitive_fields
            }
        ),
        "reduced_metric_reconstruction_maximum_absolute": float(max(
            np.max(np.abs(psi**2 * np.exp(2.0*a) - compact)),
            np.max(np.abs(psi**2 * np.exp(2.0*b) - radial)),
            np.max(np.abs(psi**2 * np.exp(2.0*c) - transverse)),
        )),
    }


def _retained(array, buffer_points=7):
    array = np.asarray(array)
    count = int(buffer_points)
    if array.shape[0] <= 2*count or array.shape[1] <= 2*count:
        raise ValueError("array is too small for retained-interior audit")
    return array[count:-count, count:-count]


def residual_norms(residual, normalized=None, buffer_points=7):
    residual = np.asarray(residual, dtype=float)
    normalized = (
        np.abs(residual) if normalized is None
        else np.asarray(normalized, dtype=float)
    )
    retained_raw = _retained(residual, buffer_points)
    retained_normalized = _retained(normalized, buffer_points)
    return {
        "retained_raw_RMS": float(np.sqrt(np.mean(retained_raw**2))),
        "retained_raw_Linf": float(np.max(np.abs(retained_raw))),
        "retained_normalized_RMS": float(np.sqrt(np.mean(
            retained_normalized**2
        ))),
        "retained_normalized_Linf": float(np.max(np.abs(
            retained_normalized
        ))),
        "global_raw_Linf": float(np.max(np.abs(residual))),
        "global_normalized_Linf": float(np.max(np.abs(normalized))),
        "finite": bool(
            np.all(np.isfinite(residual))
            and np.all(np.isfinite(normalized))
        ),
    }


def raw_hamiltonian_audit(projected, mass_squared, buffer_points=7):
    fields = reconstruct_anisotropic_fields(projected)
    residual = anisotropic_hamiltonian_residual(
        projected.z, projected.r, fields["psi"], fields["a"], fields["b"],
        fields["c"], fields["phi"], fields["chi_r"], fields["chi_z"],
        mass_squared, chi=fields["chi"], stencil_width=STENCIL_WIDTH,
    )
    geometry = axisymmetric_diagonal_geometry(
        projected.z, projected.r, fields["psi"], fields["a"], fields["b"],
        fields["c"], STENCIL_WIDTH,
    )
    phi = fields["phi"]
    # This explicit scale retains the unbalanced raw Hamiltonian as a
    # diagnostic.  The balanced constraint below is the prospective gate.
    curvature = geometry["scalar_curvature"]
    potential = float(mass_squared) * phi**2
    scale = np.maximum(1.0, np.abs(curvature) + 12.0 + np.abs(potential))
    return {
        **residual_norms(residual, np.abs(residual)/scale, buffer_points),
        "normalization": "max(1,|R|+12+|m_phi^2 Phi^2|)",
        "tracefree_maximum_absolute": fields["tracefree_maximum_absolute"],
        "reconstruction_maximum_absolute": fields[
            "reduced_metric_reconstruction_maximum_absolute"
        ],
    }


def balanced_constraint_audit(projected, background, reference, buffer_points=7):
    """Evaluate the balanced constraint without solving projected q or Phi."""
    fields = reconstruct_anisotropic_fields(projected)
    conformal_q = 1.0 / fields["psi"] - projected.z[:, None]
    arguments = (
        conformal_q, fields["phi"], projected.z, projected.r,
        fields["a"], fields["b"], fields["c"], background,
        fields["chi_r"], fields["chi_z"],
        np.asarray(reference["q"]), np.asarray(reference["phi"]),
    )
    balanced = anisotropic_initial_data_residual(
        *arguments, stencil_width=STENCIL_WIDTH,
    )
    raw = _raw_residual_and_jacobian(
        *arguments, STENCIL_WIDTH, False,
    )
    zero = np.zeros_like(fields["a"])
    defect = _raw_residual_and_jacobian(
        np.asarray(reference["q"]), np.asarray(reference["phi"]),
        projected.z, projected.r, zero, zero, zero, background,
        fields["chi_r"], fields["chi_z"],
        np.asarray(reference["q"]), np.asarray(reference["phi"]),
        STENCIL_WIDTH, False,
    )
    if not np.allclose(balanced, raw-defect, rtol=0.0, atol=1e-12):
        raise RuntimeError("balanced residual decomposition mismatch")
    shape = (len(projected.z), len(projected.r))
    count = int(np.prod(shape))
    scale = np.maximum(1.0, np.abs(raw) + np.abs(defect))
    records = {}
    for name, section in (
        ("hamiltonian", slice(0, count)),
        ("scalar", slice(count, 2*count)),
    ):
        value = balanced[section].reshape(shape)
        norm = (np.abs(balanced[section])/scale[section]).reshape(shape)
        records[name] = residual_norms(value, norm, buffer_points)
    return {
        "rows": records,
        "combined_retained_normalized_RMS": float(np.sqrt(np.mean(np.concatenate([
            _retained(
                (np.abs(balanced[section])/scale[section]).reshape(shape),
                buffer_points,
            ).ravel()
            for section in (slice(0, count), slice(count, 2*count))
        ])**2))),
        "combined_retained_normalized_Linf": max(
            record["retained_normalized_Linf"] for record in records.values()
        ),
        "normalization": "max(1,|raw shaped equation|+|reference defect|)",
        "projected_fields_solved": False,
        "finite": bool(all(record["finite"] for record in records.values())),
    }


def lorentzian_position_signature(z, r, reduced_fields):
    """Vectorized signature audit for one regular reduced position field."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(reduced_fields, dtype=float)
    if q.shape != (len(z), len(r), FIELD_COUNT):
        raise ValueError("signature position field has the wrong shape")
    metric = np.zeros((*q.shape[:2], 5, 5))
    radius = r[None, :]
    metric[:, :, 0, 0] = q[:, :, 2]
    metric[:, :, 0, 1] = metric[:, :, 1, 0] = q[:, :, 0]
    metric[:, :, 0, 2] = metric[:, :, 2, 0] = radius*q[:, :, 5]
    metric[:, :, 1, 1] = q[:, :, 6]
    metric[:, :, 1, 2] = metric[:, :, 2, 1] = radius*q[:, :, 1]
    metric[:, :, 2, 2] = q[:, :, 3]+radius**2*q[:, :, 4]
    metric[:, :, 3, 3] = q[:, :, 3]
    metric[:, :, 4, 4] = q[:, :, 3]
    eigenvalues = np.linalg.eigvalsh(metric)
    negative_counts = np.count_nonzero(eigenvalues < 0.0, axis=-1)
    margins = np.min(np.abs(eigenvalues), axis=-1)/np.maximum(
        np.max(np.abs(eigenvalues), axis=-1), 1e-300,
    )
    return {
        "all_points_one_negative_direction": bool(all(
            value == 1 for value in negative_counts.ravel()
        )),
        "minimum_eigenvalue_margin": float(np.min(margins)),
        "minimum_negative_count": int(np.min(negative_counts)),
        "maximum_negative_count": int(np.max(negative_counts)),
        "finite": bool(np.all(np.isfinite(margins))),
        "point_count": int(np.prod(q.shape[:2])),
    }


def lorentzian_signature_audit(projected):
    return lorentzian_position_signature(
        projected.z, projected.r, projected.reduced_fields,
    )


def analytic_endpoint_wall_audit(projected, background, radial_buffer=7):
    """Score the clamped analytic wall derivatives before production FD rows."""
    q = projected.reduced_fields
    v = projected.reduced_first[0]
    qz = projected.reduced_first[1]
    vz = projected.reduced_second[0, 1]
    r = projected.r
    radius = r[None, :]
    radius2 = radius**2

    def components(state):
        return {
            "tt": state[:, :, 2],
            "rr": state[:, :, 3] + radius2*state[:, :, 4],
            "sphere": state[:, :, 3],
            "tr": radius*state[:, :, 5],
        }

    q_component = components(q)
    v_component = components(v)
    qz_component = components(qz)
    vz_component = components(vz)
    walls = {}
    combined_position = []
    combined_tangent = []
    retained_slice = (
        slice(None, -int(radial_buffer)) if radial_buffer else slice(None)
    )
    for wall in ("lower", "upper"):
        index = 0 if wall == "lower" else -1
        source = wall_source_coefficients(q[index, :, 7], background, wall)
        beta = source["beta"]
        beta_t = source["beta_phi"]*v[index, :, 7]
        A = np.sqrt(q[index, :, 6])
        A_t = v[index, :, 6]/(2.0*A)
        metric_profiles = {}
        metric_position_norms = []
        metric_tangent_norms = []
        for name in ("tt", "rr", "sphere", "tr"):
            value = q_component[name][index]
            value_t = v_component[name][index]
            derivative = qz_component[name][index]
            derivative_t = vz_component[name][index]
            source_term = 2.0*beta*A*value
            source_tangent = 2.0*(
                beta_t*A*value + beta*A_t*value + beta*A*value_t
            )
            row = derivative + source_term
            tangent = derivative_t + source_tangent
            normalized = np.abs(row)/np.maximum(
                1.0, np.abs(derivative)+np.abs(source_term),
            )
            tangent_normalized = np.abs(tangent)/np.maximum(
                1.0, np.abs(derivative_t)+np.abs(source_tangent),
            )
            metric_position_norms.append(normalized)
            metric_tangent_norms.append(tangent_normalized)
            metric_profiles[name] = {
                "row": row.tolist(), "normalized": normalized.tolist(),
                "tangent_row": tangent.tolist(),
                "tangent_normalized": tangent_normalized.tolist(),
            }
        metric_position = np.maximum.reduce(metric_position_norms)
        metric_tangent = np.maximum.reduce(metric_tangent_norms)
        sign = float(source["orientation"])
        gamma = float(background["wall_stiffness"])
        delta = q[index, :, 7]-float(source["target"])
        phi_source = sign*0.5*gamma*delta*A
        phi_source_t = sign*0.5*gamma*(v[index, :, 7]*A+delta*A_t)
        phi_row = qz[index, :, 7]+phi_source
        phi_tangent = vz[index, :, 7]+phi_source_t
        phi_normalized = np.abs(phi_row)/np.maximum(
            1.0, np.abs(qz[index, :, 7])+np.abs(phi_source),
        )
        phi_tangent_normalized = np.abs(phi_tangent)/np.maximum(
            1.0, np.abs(vz[index, :, 7])+np.abs(phi_source_t),
        )
        chi_row = qz[index, :, 8]
        chi_tangent = vz[index, :, 8]
        chi_normalized = np.abs(chi_row)/np.maximum(1.0, np.abs(chi_row))
        chi_tangent_normalized = np.abs(chi_tangent)/np.maximum(
            1.0, np.abs(chi_tangent),
        )
        gauge_position = np.maximum(
            np.abs(q[index, :, 0]), np.abs(r*q[index, :, 1]),
        )
        gauge_tangent = np.maximum(
            np.abs(v[index, :, 0]), np.abs(r*v[index, :, 1]),
        )
        position_stack = np.concatenate([
            metric_position[retained_slice], phi_normalized[retained_slice],
            chi_normalized[retained_slice], gauge_position[retained_slice],
        ])
        tangent_stack = np.concatenate([
            metric_tangent[retained_slice],
            phi_tangent_normalized[retained_slice],
            chi_tangent_normalized[retained_slice],
            gauge_tangent[retained_slice],
        ])
        combined_position.append(position_stack)
        combined_tangent.append(tangent_stack)
        walls[wall] = {
            "metric": metric_profiles,
            "Phi": {
                "row": phi_row.tolist(), "normalized": phi_normalized.tolist(),
                "tangent_row": phi_tangent.tolist(),
                "tangent_normalized": phi_tangent_normalized.tolist(),
            },
            "chi": {
                "row": chi_row.tolist(), "normalized": chi_normalized.tolist(),
                "tangent_row": chi_tangent.tolist(),
                "tangent_normalized": chi_tangent_normalized.tolist(),
            },
            "gauge_position": gauge_position.tolist(),
            "gauge_tangent": gauge_tangent.tolist(),
            "position_normalized_RMS": float(np.sqrt(np.mean(position_stack**2))),
            "position_normalized_Linf": float(np.max(position_stack)),
            "tangent_normalized_RMS": float(np.sqrt(np.mean(tangent_stack**2))),
            "tangent_normalized_Linf": float(np.max(tangent_stack)),
        }
    position = np.concatenate(combined_position)
    tangent = np.concatenate(combined_tangent)
    return {
        "walls": walls,
        "position_normalized_RMS": float(np.sqrt(np.mean(position**2))),
        "position_normalized_Linf": float(np.max(position)),
        "tangent_normalized_RMS": float(np.sqrt(np.mean(tangent**2))),
        "tangent_normalized_Linf": float(np.max(tangent)),
        "finite": bool(np.all(np.isfinite(position)) and np.all(np.isfinite(tangent))),
        "scope": "analytic clamped endpoint derivatives before target finite differences",
    }


def normal_gauge_position_audit(
    projected, source, background, radial_buffer=7, analytic_derivative=False,
):
    """Retain signed and normalized normal-GH position-wall profiles."""
    q = projected.reduced_fields
    z = projected.z
    r = projected.r
    h = np.asarray(source, dtype=float)
    dz = derivative_matrix(z, 1, STENCIL_WIDTH)
    if hasattr(dz, "toarray"):
        dz = dz.toarray()
    retained = (
        slice(None, -int(radial_buffer)) if radial_buffer else slice(None)
    )
    records = {}
    combined = []
    for wall, index in (("lower", 0), ("upper", -1)):
        wall_source = wall_source_coefficients(q[index, :, 7], background, wall)
        G = q[:, :, 6]
        derivative = (
            projected.reduced_first[1, index, :, 6]
            if analytic_derivative else (dz@G)[index]
        )
        terms = (
            derivative,
            8.0*wall_source["beta"]*G[index]**1.5,
            -2.0*h[index, :, 1]*G[index],
        )
        row = sum(terms)
        scale = np.maximum(1.0, sum(np.abs(term) for term in terms))
        normalized = np.abs(row)/scale
        local = normalized[retained]
        combined.append(local)
        records[wall] = {
            "row": row.tolist(), "normalized": normalized.tolist(),
            "normalized_RMS": float(np.sqrt(np.mean(local**2))),
            "normalized_Linf": float(np.max(local)),
        }
    values = np.concatenate(combined)
    return {
        "walls": records,
        "combined_normalized_RMS": float(np.sqrt(np.mean(values**2))),
        "combined_normalized_Linf": float(np.max(values)),
        "finite": bool(np.all(np.isfinite(values))),
        "derivative_route": (
            "analytic clamped q_z" if analytic_derivative
            else "production seven-point finite difference"
        ),
    }


def wall_junction_audit(projected, background, radial_buffer=7):
    walls = {}
    all_normalized = []
    scored = (
        slice(None, -int(radial_buffer)) if radial_buffer else slice(None)
    )
    for wall in ("lower", "upper"):
        record = wall_junction_rows(
            projected.reduced_fields, projected.reduced_first[0],
            projected.z, projected.r, background, wall, STENCIL_WIDTH,
        )
        component_profiles = [
            value["robin_normalized"] for value in record["components"].values()
        ]
        metric_profile = np.maximum.reduce(component_profiles)
        retained = metric_profile[scored]
        scalar = record["separate_rows"]
        wall_index = 0 if wall == "lower" else -1
        orientation = float(record["orientation"])
        gamma = float(background["wall_stiffness"])
        A = np.sqrt(projected.reduced_fields[wall_index, :, 6])
        delta = (
            projected.reduced_fields[wall_index, :, 7]
            - float(record["source"]["target"])
        )
        phi_source = orientation*0.5*gamma*delta*A
        phi_derivative = scalar["Phi_robin"]-phi_source
        phi_scale = np.maximum(
            1.0, np.abs(phi_derivative)+np.abs(phi_source),
        )
        phi_profile = np.abs(scalar["Phi_robin"])/phi_scale
        chi_profile = np.abs(scalar["chi_neumann"])/np.maximum(
            1.0, np.abs(scalar["chi_neumann"]),
        )
        all_normalized.extend((retained, phi_profile[scored], chi_profile[scored]))
        walls[wall] = {
            "metric_normalized_RMS": float(np.sqrt(np.mean(retained**2))),
            "metric_normalized_Linf": float(np.max(retained)),
            "Phi_normalized_RMS": float(np.sqrt(np.mean(phi_profile[scored]**2))),
            "Phi_normalized_Linf": float(np.max(phi_profile[scored])),
            "chi_normalized_RMS": float(np.sqrt(np.mean(chi_profile[scored]**2))),
            "chi_normalized_Linf": float(np.max(chi_profile[scored])),
            "J_orthonormal_Linf": float(np.max(
                record["tensor_norms"]["J_orthonormal_frobenius"][scored]
            )),
            "DXJ_orthonormal_Linf": float(np.max(
                record["tensor_norms"]["DXJ_orthonormal_frobenius"][scored]
            )),
            "profiles": {
                "metric_components": {
                    name: {
                        "robin_residual": value["robin_residual"].tolist(),
                        "robin_normalized": value["robin_normalized"].tolist(),
                        "J": value["J"].tolist(),
                        "DXJ": value["DXJ"].tolist(),
                    }
                    for name, value in record["components"].items()
                },
                "J_tensor": record["J_tensor"].tolist(),
                "DXJ_tensor": record["DXJ_tensor"].tolist(),
                "Phi_robin": scalar["Phi_robin"].tolist(),
                "DX_Phi_robin": scalar["DX_Phi_robin"].tolist(),
                "chi_neumann": scalar["chi_neumann"].tolist(),
                "DX_chi_neumann": scalar["DX_chi_neumann"].tolist(),
                "Phi_normalized": phi_profile.tolist(),
                "chi_normalized": chi_profile.tolist(),
            },
            "finite": bool(record["finite"]),
        }
    combined = np.concatenate([np.ravel(value) for value in all_normalized])
    return {
        "walls": walls,
        "combined_normalized_RMS": float(np.sqrt(np.mean(combined**2))),
        "combined_normalized_Linf": float(np.max(combined)),
        "finite": bool(all(item["finite"] for item in walls.values())),
    }


def projected_second_wall_audit(projected, background, radial_buffer=7):
    """Score the directly projected parent q0,v0,a0 before any RHS closure."""
    walls = {}
    combined = []
    scored = (
        slice(None, -int(radial_buffer)) if radial_buffer else slice(None)
    )
    for wall in ("lower", "upper"):
        record = wall_junction_second_tangent(
            projected.reduced_fields, projected.reduced_first[0],
            projected.reduced_second[0, 0], projected.z, projected.r,
            background, wall, STENCIL_WIDTH,
        )
        frames, frame_defect = _orthonormal_frames(record["metric_tensor"])

        def orthonormal(key):
            return np.einsum(
                "nai,nab,nbj->nij", frames, record[key], frames,
            )

        metric_hat = orthonormal("DX2J_tensor")
        linear_hat = orthonormal("DJ_acceleration_tensor")
        hessian_hat = orthonormal("D2J_velocity_velocity_tensor")
        metric = np.linalg.norm(metric_hat, axis=(1, 2))
        linear = np.linalg.norm(linear_hat, axis=(1, 2))
        hessian = np.linalg.norm(hessian_hat, axis=(1, 2))
        normalized_metric = metric/(1.0+linear+hessian)
        separate = record["separate_rows"]
        phi = np.abs(separate["DX2_Phi_robin"])
        phi_scale = (
            1.0 + np.abs(separate["DJ_Phi_robin_acceleration"])
            + np.abs(separate["D2_Phi_robin_velocity_velocity"])
        )
        chi = np.abs(separate["DX2_chi_neumann"])
        chi_scale = 1.0 + np.abs(separate["DJ_chi_neumann_acceleration"])
        profiles = {
            "metric": normalized_metric[scored],
            "Phi": (phi/phi_scale)[scored],
            "chi": (chi/chi_scale)[scored],
        }
        combined.extend(profiles.values())
        walls[wall] = {
            name: {
                "normalized_RMS": float(np.sqrt(np.mean(profile**2))),
                "normalized_Linf": float(np.max(profile)),
            } for name, profile in profiles.items()
        }
        walls[wall]["decomposition_maximum_absolute_defect"] = float(
            record["decomposition_maximum_absolute_defect"]
        )
        walls[wall]["frame_defect_maximum"] = float(np.max(frame_defect))
        walls[wall]["profiles"] = {
            "DX2J_tensor": record["DX2J_tensor"].tolist(),
            "DJ_acceleration_tensor": record["DJ_acceleration_tensor"].tolist(),
            "D2J_velocity_velocity_tensor": record[
                "D2J_velocity_velocity_tensor"
            ].tolist(),
            "DX2J_orthonormal_frobenius": metric.tolist(),
            "DJ_acceleration_orthonormal_frobenius": linear.tolist(),
            "D2J_velocity_velocity_orthonormal_frobenius": hessian.tolist(),
            "metric_normalized": normalized_metric.tolist(),
            "Phi_DX2": separate["DX2_Phi_robin"].tolist(),
            "Phi_DJ_acceleration": separate[
                "DJ_Phi_robin_acceleration"
            ].tolist(),
            "Phi_D2_velocity_velocity": separate[
                "D2_Phi_robin_velocity_velocity"
            ].tolist(),
            "Phi_normalized": (phi/phi_scale).tolist(),
            "chi_DX2": separate["DX2_chi_neumann"].tolist(),
            "chi_DJ_acceleration": separate[
                "DJ_chi_neumann_acceleration"
            ].tolist(),
            "chi_normalized": (chi/chi_scale).tolist(),
        }
        walls[wall]["finite"] = bool(
            record["finite"] and np.all(np.isfinite(frame_defect))
            and np.all(np.isfinite(normalized_metric))
        )
    combined_array = np.concatenate([np.ravel(value) for value in combined])
    return {
        "walls": walls,
        "combined_normalized_RMS": float(np.sqrt(np.mean(combined_array**2))),
        "combined_normalized_Linf": float(np.max(combined_array)),
        "finite": bool(all(record["finite"] for record in walls.values())),
        "scope": "direct P11 projection before target RHS boundary operations",
    }


def second_wall_closure_audit(
    position, velocity, acceleration, z, r, background, radial_buffer=0,
):
    """Normalized twice-differentiated metric/Phi/chi closure profiles."""
    retained = (
        slice(None, -int(radial_buffer)) if radial_buffer else slice(None)
    )
    walls = {}
    combined = []
    for wall in ("lower", "upper"):
        record = wall_junction_second_tangent(
            position, velocity, acceleration, z, r, background, wall,
            STENCIL_WIDTH,
        )
        metric_profiles = {}
        metric_normalized = []
        for name, component in record["components"].items():
            row = component["DX2_robin_residual"]
            derivative = component["normal_derivative_tt"]
            source = row-derivative
            normalized = np.abs(row)/np.maximum(
                1.0, np.abs(derivative)+np.abs(source),
            )
            metric_normalized.append(normalized)
            metric_profiles[name] = {
                "row": row.tolist(), "normalized": normalized.tolist(),
            }
        metric = np.maximum.reduce(metric_normalized)
        separate = record["separate_rows"]
        phi_row = separate["DX2_Phi_robin"]
        phi_scale = np.maximum(
            1.0,
            np.abs(separate["DJ_Phi_robin_acceleration"])
            + np.abs(separate["D2_Phi_robin_velocity_velocity"]),
        )
        phi = np.abs(phi_row)/phi_scale
        chi_row = separate["DX2_chi_neumann"]
        chi = np.abs(chi_row)/np.maximum(1.0, np.abs(chi_row))
        local = np.concatenate((
            metric[retained], phi[retained], chi[retained],
        ))
        combined.append(local)
        walls[wall] = {
            "metric": metric_profiles,
            "Phi": {"row": phi_row.tolist(), "normalized": phi.tolist()},
            "chi": {"row": chi_row.tolist(), "normalized": chi.tolist()},
            "combined_normalized_RMS": float(np.sqrt(np.mean(local**2))),
            "combined_normalized_Linf": float(np.max(local)),
            "finite": bool(record["finite"] and np.all(np.isfinite(local))),
        }
    values = np.concatenate(combined)
    return {
        "walls": walls,
        "combined_normalized_RMS": float(np.sqrt(np.mean(values**2))),
        "combined_normalized_Linf": float(np.max(values)),
        "finite": bool(all(item["finite"] for item in walls.values())),
        "radial_buffer": int(radial_buffer),
    }


@dataclass(frozen=True)
class MatchedCaseBundle:
    label: str
    geometry: dict
    driver: DriverConfiguration
    taylor: object
    taylor_fingerprint: str
    source0: np.ndarray
    memory0: np.ndarray
    source_time0: np.ndarray
    source_second_time0: np.ndarray
    outer_reference_position: np.ndarray
    outer_reference_acceleration: np.ndarray
    rhs_by_mode: dict
    common_input_fingerprint: str
    configuration: dict

    def __post_init__(self):
        for name in (
            "source0", "memory0", "source_time0", "source_second_time0",
            "outer_reference_position", "outer_reference_acceleration",
        ):
            object.__setattr__(self, name, _immutable_array(
                getattr(self, name), float,
            ))
        object.__setattr__(self, "rhs_by_mode", MappingProxyType(dict(
            self.rhs_by_mode
        )))
        object.__setattr__(self, "geometry", _freeze_value(self.geometry))
        object.__setattr__(self, "configuration", _freeze_value(
            self.configuration
        ))
        object.__setattr__(self, "common_input_fingerprint", str(
            self.common_input_fingerprint
        ))
        object.__setattr__(self, "taylor_fingerprint", str(
            self.taylor_fingerprint
        ))

    def initial_state(self):
        initial = np.asarray(self.geometry["jet_field"].reduced_fields)
        velocity = np.asarray(self.geometry["jet_field"].reduced_first[0])
        return (
            initial.copy(), velocity.copy(), self.source0.copy(),
            self.memory0.copy(),
        )


def _json_bytes(value):
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
    ).encode()


def _taylor_fingerprint(taylor):
    digest = hashlib.sha256()
    for name in (
        "source_value", "source_first", "z", "r", "source_reduced",
        "source_time_reduced", "source_time_z", "source_time_r",
    ):
        _update_digest(digest, name, getattr(taylor, name))
    return digest.hexdigest()


def extended_case_fingerprint(bundle, mode, dt):
    if mode not in BOUNDARY_MODES:
        raise ValueError("unknown boundary mode")
    digest = hashlib.sha256()
    digest.update(bundle.common_input_fingerprint.encode())
    digest.update(str(mode).encode())
    digest.update(np.asarray(float(dt), dtype=np.float64).tobytes())
    return digest.hexdigest()


def build_mode_neutral_case(
    geometry, label, driver=DriverConfiguration(), provenance=None,
):
    """Construct both boundary modes from identical projected P11 inputs.

    The outer reference is the directly projected parent ``q0,a0``.  No RHS
    call is allowed to manufacture or repair it.
    """
    jet = geometry["jet_field"]
    z = np.asarray(jet.z, dtype=float)
    r = np.asarray(jet.r, dtype=float)
    if not (
        np.array_equal(z, np.asarray(geometry["z"], dtype=float))
        and np.array_equal(r, np.asarray(geometry["r"], dtype=float))
    ):
        raise ValueError("case geometry coordinates differ from projected jets")
    production = bool(
        isinstance(provenance, Mapping) and provenance.get("protocol")
    )
    if production:
        required_fingerprints = (
            "continuous_parent_fingerprint", "primitive_parent_fingerprint",
            "projection_fingerprint",
        )
        if any(
            not isinstance(geometry.get(name), str) or not geometry.get(name)
            for name in required_fingerprints
        ):
            raise ValueError("production case lacks bound parent/projection fingerprints")
        if geometry["projection_fingerprint"] != projection_fingerprint(jet):
            raise ValueError("production case projection fingerprint mismatch")
        matching = [
            spec for spec in TARGET_GRIDS.values()
            if (len(z), len(r)) == (spec.nz, spec.nr_r10)
        ]
        if len(matching) != 1:
            raise ValueError("production case is not a frozen target grid")
        projection_metadata = geometry.get("projection", {})
        if projection_metadata.get("grid_label") != matching[0].label:
            raise ValueError("production case target label does not match its grid")
        expected_z, expected_r = target_coordinates(
            np.linspace(1.0, np.e, 145), matching[0],
        )
        if not (np.array_equal(z, expected_z) and np.array_equal(r, expected_r)):
            raise ValueError("production target coordinates are not canonical")
    initial = np.asarray(jet.reduced_fields, dtype=float)
    velocity = np.asarray(jet.reduced_first[0], dtype=float)
    acceleration0 = np.asarray(jet.reduced_second[0, 0], dtype=float)
    expected = (len(z), len(r), FIELD_COUNT)
    if initial.shape != expected or velocity.shape != expected or acceleration0.shape != expected:
        raise ValueError("projected case arrays have invalid shapes")
    taylor = gauge_taylor_source_from_initial_jets(jet, z, r)
    for name in (
        "source_value", "source_first", "z", "r", "source_reduced",
        "source_time_reduced", "source_time_z", "source_time_r",
    ):
        setattr(taylor, name, _immutable_array(getattr(taylor, name), float))
    taylor_fingerprint = _taylor_fingerprint(taylor)
    source = taylor.source_reduced.copy()
    source_time = taylor.source_time_reduced.copy()
    target = regular_so3_nonlinear_anchored_damped_wave_target(
        initial, initial, source, r, driver.target_mu_lapse,
        driver.target_mu_shift, driver.target_power,
    )
    source_z, source_r = regular_source_spatial_derivatives(
        source, z, r, driver.stencil_width,
    )
    advection = regular_so3_live_source_shift_advection(
        initial, r, source, source_z, source_r,
    )
    memory = (
        source_time - advection + driver.driver_mu * (source-target)
    )
    source_dot, memory_dot = source_driver_rhs(
        source, memory, target, driver.driver_mu, driver.driver_eta, advection,
    )
    source_second = live_regular_source_second_time(
        initial, velocity, initial, source, source, source_dot, memory_dot,
        z, r, driver.driver_mu, driver.target_mu_lapse,
        driver.target_mu_shift, driver.target_power, driver.stencil_width,
    )
    normal = np.stack((acceleration0[0, :, 6], acceleration0[-1, :, 6]))
    outer_position = initial.copy()
    outer_acceleration = acceleration0.copy()
    rhs_by_mode = {}
    for mode in BOUNDARY_MODES:
        rhs = NativeRegularSO3RHS(
            z, r, taylor, geometry["mass_squared"], geometry["background"],
            normal, stencil_width=driver.stencil_width,
            live_normal_wall_gauge=True, live_outer_sommerfeld=True,
            boundary_closure_mode=mode,
        )
        rhs.set_outer_sommerfeld_reference(
            outer_position, outer_acceleration,
        )
        rhs_by_mode[mode] = rhs
    configuration = {
        "label": str(label),
        "grid_shape": list(expected),
        "mass_squared": float(geometry["mass_squared"]),
        "background": _jsonable(geometry["background"]),
        "driver": driver.public(),
        "boundary_modes": list(BOUNDARY_MODES),
        "live_normal_wall_gauge": True,
        "live_outer_sommerfeld": True,
        "outer_reference_origin": "direct projected P11 q0,a0; no RHS construction",
        "projection": geometry.get("projection", {}),
        "continuous_parent_fingerprint": geometry.get(
            "continuous_parent_fingerprint"
        ),
        "primitive_parent_fingerprint": geometry.get(
            "primitive_parent_fingerprint"
        ),
        "projection_fingerprint": geometry.get("projection_fingerprint"),
        "provenance": _jsonable(provenance or {}),
    }
    common = hashlib.sha256()
    immutable_inputs = {
        "z": z,
        "r": r,
        "reduced_fields": initial,
        "reduced_first": jet.reduced_first,
        "reduced_second": jet.reduced_second,
        "primitive_fields": jet.primitive_fields or {},
        "source0": source,
        "memory0": memory,
        "source_time0": source_time,
        "source_second_time0": source_second,
        "taylor_source_value": taylor.source_value,
        "taylor_source_first": taylor.source_first,
        "taylor_fingerprint": taylor_fingerprint,
        "outer_reference_position": outer_position,
        "outer_reference_acceleration": outer_acceleration,
        "configuration": configuration,
    }
    for name, value in immutable_inputs.items():
        _update_digest(common, name, value)
    return MatchedCaseBundle(
        str(label), geometry, driver, taylor, taylor_fingerprint,
        source, memory, source_time,
        source_second, outer_position, outer_acceleration, rhs_by_mode,
        common.hexdigest(), configuration,
    )


def validate_bundle_integrity(bundle, mode=None):
    """Fail closed if a live RHS differs from the content-bound case."""
    modes = BOUNDARY_MODES if mode is None else (mode,)
    expected_normal = np.stack((
        bundle.outer_reference_acceleration[0, :, 6],
        bundle.outer_reference_acceleration[-1, :, 6],
    ))
    checks = {}
    for selected in modes:
        if selected not in BOUNDARY_MODES:
            raise ValueError("unknown boundary mode")
        rhs = bundle.rhs_by_mode[selected]
        local = {
            "mode": rhs.boundary_closure_mode == selected,
            "z": np.array_equal(rhs.z, bundle.geometry["jet_field"].z),
            "r": np.array_equal(rhs.r, bundle.geometry["jet_field"].r),
            "normal": np.array_equal(rhs.normal_wall_acceleration, expected_normal),
            "outer_position": np.array_equal(
                rhs.outer_reference_position, bundle.outer_reference_position,
            ),
            "outer_acceleration": np.array_equal(
                rhs.outer_reference_acceleration,
                bundle.outer_reference_acceleration,
            ),
            "background": _json_bytes(rhs.background) == _json_bytes(
                bundle.geometry["background"]
            ),
            "mass": rhs.mass_squared == float(bundle.geometry["mass_squared"]),
            "stencil": rhs.stencil_width == bundle.driver.stencil_width,
            "live_normal": bool(rhs.live_normal_wall_gauge),
            "live_outer": bool(rhs.live_outer_sommerfeld),
            "taylor_identity": rhs.gauge_source is bundle.taylor,
            "taylor_content": (
                _taylor_fingerprint(bundle.taylor)
                == bundle.taylor_fingerprint
            ),
        }
        checks[selected] = local
        if not all(local.values()):
            failed = [name for name, passed in local.items() if not passed]
            raise RuntimeError(
                f"sealed case/RHS integrity failure for {selected}: {failed}"
            )
    return checks


def reconstruct_driver_stage(bundle, mode, time_value, state, capture=True):
    """Return one current-code RHS stage and every provenance-critical array."""
    if mode not in BOUNDARY_MODES:
        raise ValueError("unknown boundary mode")
    validate_bundle_integrity(bundle, mode)
    position, velocity, source, memory = (
        np.asarray(value, dtype=float) for value in state
    )
    z = bundle.geometry["z"]
    r = bundle.geometry["r"]
    config = bundle.driver
    source_z, source_r = regular_source_spatial_derivatives(
        source, z, r, config.stencil_width,
    )
    target = regular_so3_nonlinear_anchored_damped_wave_target(
        position, bundle.geometry["jet_field"].reduced_fields,
        bundle.source0, r, config.target_mu_lapse,
        config.target_mu_shift, config.target_power,
    )
    advection = regular_so3_live_source_shift_advection(
        position, r, source, source_z, source_r,
    )
    source_time, memory_time = source_driver_rhs(
        source, memory, target, config.driver_mu, config.driver_eta, advection,
    )
    source_time, outer_source = apply_outer_source_sommerfeld(
        source, source_time, bundle.source0, bundle.source_time0,
        bundle.source_second_time0, position, time_value, r,
        config.stencil_width,
    )
    gauge = StageRegularGaugeSource(
        source, source_time, z, r, config.stencil_width,
    )
    source_second = live_regular_source_second_time(
        position, velocity, bundle.geometry["jet_field"].reduced_fields,
        bundle.source0, source, source_time, memory_time, z, r,
        config.driver_mu, config.target_mu_lapse, config.target_mu_shift,
        config.target_power, config.stencil_width,
    )
    acceleration, diagnostic = bundle.rhs_by_mode[mode].acceleration(
        time_value, position, velocity, gauge, source_second,
        capture_boundary_stages=bool(capture),
    )
    stages = diagnostic["boundary_stages"] if capture else None
    landmarks = {} if stages is None else select_landmarks(mode, stages)
    preservation = None if stages is None else evaluate_boundary_stage_sequence(
        position, velocity, stages, z, r, bundle.geometry["background"],
        config.stencil_width, buffer_points=config.stencil_width,
    )
    finite = bool(all(np.all(np.isfinite(value)) for value in (
        position, velocity, source, memory, source_time, memory_time, target,
        advection, source_second, acceleration,
    )) and diagnostic["finite"] and (
        stages is None or all(np.all(np.isfinite(stage["acceleration"])) for stage in stages)
    ) and (preservation is None or preservation["finite"]))
    return {
        "time": float(time_value),
        "mode": mode,
        "q": position.copy(), "v": velocity.copy(),
        "source": source.copy(), "memory": memory.copy(),
        "source_time": source_time.copy(),
        "memory_time": memory_time.copy(),
        "target": target.copy(), "advection": advection.copy(),
        "source_second_time": source_second.copy(),
        "acceleration": acceleration.copy(),
        "slopes": (
            velocity.copy(), acceleration.copy(), source_time.copy(),
            memory_time.copy(),
        ),
        "boundary_stages": stages,
        "landmarks": landmarks,
        "staged_preservation": preservation,
        "diagnostic": diagnostic,
        "outer_source_diagnostic": outer_source,
        "finite": finite,
    }


def select_landmarks(mode, stages):
    required = MANDATORY_LANDMARKS[mode]
    names = [str(stage["name"]) for stage in stages]
    selected = {}
    indices = []
    for name in required:
        matches = [index for index, found in enumerate(names) if found == name]
        if len(matches) != 1:
            raise RuntimeError(f"mandatory stage {name} is not unique")
        index = matches[0]
        indices.append(index)
        selected[name] = np.asarray(stages[index]["acceleration"]).copy()
    if indices != sorted(indices):
        raise RuntimeError("mandatory boundary landmarks are out of order")
    return selected


def axis_even_crossfit_audit(
    position, acceleration, r, widths=(0.40, 0.50, 0.60), degree=3,
):
    """Compare degree-three even-axis coefficient jets across proper widths."""
    position = np.asarray(position, dtype=float)
    values = np.asarray(acceleration, dtype=float)
    r = np.asarray(r, dtype=float)
    if (
        values.ndim != 3 or values.shape[1] != len(r)
        or values.shape[2] != FIELD_COUNT or position.shape != values.shape
    ):
        raise ValueError("axis cross-fit acceleration has the wrong shape")
    radial_metric = position[:, :, 3]+r[None, :]**2*position[:, :, 4]
    if np.any(radial_metric <= 0.0):
        raise ValueError("axis cross-fit radial metric is not positive")
    proper = np.zeros_like(radial_metric)
    increments = 0.5*(
        np.sqrt(radial_metric[:, 1:])+np.sqrt(radial_metric[:, :-1])
    )*np.diff(r)[None, :]
    proper[:, 1:] = np.cumsum(increments, axis=1)
    fits = np.empty((len(widths), values.shape[0], FIELD_COUNT, degree+1))
    counts = np.empty((len(widths), values.shape[0]), dtype=int)
    for width_index, width in enumerate(widths):
        for i in range(values.shape[0]):
            indices = np.flatnonzero(proper[i] <= float(width)+1e-14)
            if len(indices) < int(degree)+1:
                raise ValueError("axis cross-fit width has too few radial points")
            coordinate = r[indices]**2
            counts[width_index, i] = len(indices)
            for field in range(FIELD_COUNT):
                fits[width_index, i, field] = np.polynomial.polynomial.polyfit(
                    coordinate, values[i, indices, field], int(degree),
                )
    stacked = fits
    spread = np.max(stacked, axis=0)-np.min(stacked, axis=0)
    scale = np.maximum(1.0, np.max(np.abs(stacked), axis=0))
    scaled = np.abs(spread)/scale
    return {
        "widths": [float(value) for value in widths],
        "point_count_minimum_by_width": np.min(counts, axis=1).tolist(),
        "point_count_maximum_by_width": np.max(counts, axis=1).tolist(),
        "polynomial_coordinate": "r^2",
        "window_coordinate": "proper radial distance from projected metric",
        "degree": int(degree),
        "coefficients": stacked.tolist(),
        "maximum_scaled_spread": float(np.max(scaled)),
        "by_field_maximum_scaled_spread": {
            REDUCED_FIELD_ORDER[field]: float(np.max(scaled[:, field]))
            for field in range(FIELD_COUNT)
        },
        "finite": bool(np.all(np.isfinite(stacked))),
    }


def trace_observational_check(bundle, mode):
    state = bundle.initial_state()
    without = reconstruct_driver_stage(bundle, mode, 0.0, state, capture=False)
    with_trace = reconstruct_driver_stage(bundle, mode, 0.0, state, capture=True)
    independent_copies = bool(all(
        not np.shares_memory(stage["acceleration"], with_trace["acceleration"])
        for stage in with_trace["boundary_stages"]
    ))
    return {
        "acceleration_bitwise_equal": bool(np.array_equal(
            without["acceleration"], with_trace["acceleration"],
        )),
        "source_time_bitwise_equal": bool(np.array_equal(
            without["source_time"], with_trace["source_time"],
        )),
        "captured_arrays_are_independent_copies": independent_copies,
        "stage_names": [
            stage["name"] for stage in with_trace["boundary_stages"]
        ],
        "finite": bool(without["finite"] and with_trace["finite"]),
    }, with_trace


def initial_native_audit(bundle):
    projected = bundle.geometry["jet_field"]
    state = bundle.initial_state()
    source_gauge = StageRegularGaugeSource(
        bundle.source0, bundle.source_time0,
        bundle.geometry["z"], bundle.geometry["r"],
        bundle.driver.stencil_width,
    )
    constraint = gauge_constraint_summary(
        state[0], state[1], 0.0,
        bundle.rhs_by_mode["legacy_wall_axis_outer"],
        bundle.driver.radial_constraint_cut, source_gauge,
    )
    wall = compact_wall_position_residuals(
        state[0], bundle.geometry["z"], bundle.geometry["r"],
        bundle.geometry["background"], bundle.driver.stencil_width, 0,
    )
    normal = compact_wall_normal_gauge_position_residuals(
        state[0], bundle.source0, bundle.geometry["z"], bundle.geometry["r"],
        bundle.geometry["background"], bundle.driver.stencil_width, 0,
    )
    return {
        "gauge_constraint": constraint,
        "compact_wall": wall,
        "normal_gauge_wall": normal,
        "normal_gauge_wall_profiles": normal_gauge_position_audit(
            projected, bundle.source0, bundle.geometry["background"], 0,
        ),
        "analytic_normal_gauge_wall_profiles": normal_gauge_position_audit(
            projected, bundle.source0, bundle.geometry["background"],
            0, analytic_derivative=True,
        ),
        "junction": wall_junction_audit(
            projected, bundle.geometry["background"], 0,
        ),
        "analytic_endpoint_wall": analytic_endpoint_wall_audit(
            projected, bundle.geometry["background"], 0,
        ),
        "projected_second_wall": projected_second_wall_audit(
            projected, bundle.geometry["background"], 0,
        ),
        "signature": lorentzian_signature_audit(projected),
    }


def run_rk2_segment(
    bundle, mode, dt=0.000125, steps=4, stage_validator=None,
):
    """Run only one bounded explicit-midpoint recovery segment."""
    if int(steps) != 4:
        raise ValueError("bounded pilot segment is frozen at four accepted steps")
    dt = float(dt)
    if dt != 0.000125:
        raise ValueError("bounded pilot segment is frozen at D1 dt=0.000125")
    state = bundle.initial_state()
    records = []

    def validated(record):
        if stage_validator is None:
            return True
        audit = stage_validator(record)
        record["technical_audit"] = audit
        return bool(audit.get("gates", {}).get("all_technical_gates_pass"))

    def result(completed):
        return {
            "mode": mode, "dt": dt, "steps": int(steps),
            "records": records,
            "end_state": tuple(np.asarray(value).copy() for value in state),
            "finite": bool(all(record["finite"] for record in records)),
            "technical_pass": bool(completed),
            "completed": bool(completed),
        }

    for step in range(1, int(steps)+1):
        start_time = (step-1)*dt
        first = reconstruct_driver_stage(
            bundle, mode, start_time, state, capture=True,
        )
        first_record = {"step": step, "rk_stage": 1, **first}
        records.append(first_record)
        if not first["finite"]:
            raise RuntimeError("nonfinite bounded pilot first RK stage")
        if not validated(first_record):
            return result(False)
        midpoint = tuple(
            value + 0.5*dt*slope
            for value, slope in zip(state, first["slopes"])
        )
        second = reconstruct_driver_stage(
            bundle, mode, start_time + 0.5*dt, midpoint, capture=True,
        )
        second_record = {"step": step, "rk_stage": 2, **second}
        records.append(second_record)
        if not second["finite"]:
            raise RuntimeError("nonfinite bounded pilot RK stage")
        if not validated(second_record):
            return result(False)
        state = tuple(
            value + dt*slope
            for value, slope in zip(state, second["slopes"])
        )
    return result(True)
