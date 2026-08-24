# Protocol 229 v4 — Free-boundary MOTS saddle-node continuation

## Question

Does the three-grid-closed zero-to-two transition of Protocol 228 arise from
a generic free-boundary saddle-node, with a simple principal zero mode,
nonzero adjoint transversality and quadratic coefficients, and invariant
square-root branch separation?

This is a continuation experiment, not a denser root-counting scan.  It treats
the capped-surface profile and time as coupled unknowns and imposes an integral
pseudo-arclength condition.

## Immutable inputs

- The exact Protocol-228 authority, source, result, endpoint checkpoints, and
  accepted G9/G10/G11 surface profiles at `t/ell = 0.0015`.
- The exact Protocol-226/227 and inherited repaired-parent evolution,
  detector, proper-geometry, and physical-normal stability implementations.
- Exact repaired G9/G10/G11 states at `t/ell = 0.001`.

Protocol 228 must replay as
`REPAIRED-PARENT-FORMATION-CLOSURE-PASS`, with every acceptance gate true.
No parent solve, state repair, projection, source change, or equation change is
authorized.

## Continuous spacetime representation

For each grid, replay the 16 already authorized explicit-midpoint RK2 steps
from global step 32 to 48.  The replay endpoint must be bitwise identical to
the corresponding Protocol-228 checkpoint.  Within each step use the fixed
second-order dense extension

`y(theta) = y_n + dt*((theta-theta^2)*k1 + theta^2*k2)`,

where `theta=(t-t_n)/dt`, `k1` is the first RK stage and `k2` the midpoint
stage.  Only position and velocity are interpolated for the surface equation.
No independently fitted temporal interpolant is permitted.

## Pseudo-arclength continuation

Let `tau=(t-0.001)/0.0005`.  Start from the Protocol-228 outer branch at
`t=0.0015`; obtain the second anchor at `t=0.00146875` by the unchanged local
BVP initialized with that profile.  The augmented BVP unknowns are
`rho(theta)`, `rho'(theta)`, an accumulated arclength integral, and `tau`.
It enforces:

1. the exact local outgoing-expansion ODE;
2. `rho'=0` at the axis and compact wall;
3. zero initial accumulated integral;
4. orthogonality to the normalized secant predictor in the product norm
   `mean_L2(rho) + tau^2`.

Use 121 initial angular nodes, at most 6000 nodes, BVP tolerance `2e-6`, dense
profiles on 501 nodes, and at most 64 accepted continuation steps.  Every new
accepted-point attempt starts with normalized arclength step `1/64`.  If the
augmented corrector fails, its arclength residual exceeds `2e-5`, or its
unchanged Protocol-227 fixed-time refinement fails inherited admission, retry
that same point at the exact dyadic sequence `1/128, 1/256, ..., 1/4096`.
Accept the first passing attempt; exhaustion at `1/4096` is technical
`BIFURCATION-INCONCLUSIVE`.  Never enlarge a step or change a scientific gate.
Archive the accepted-point index, every attempted step and solver diagnostic,
and the accepted step.  The physical-normal principal eigenvalue is evaluated
at 65 nodes after inherited fixed-time admission.

When successive points first have opposite principal signs, refine that
arclength interval by deterministic step halving.  Each requested bracket
step is itself corrected with the same first-passing dyadic policy, with a
fixed technical floor of `1/65536`; every requested and attempted step is
archived.  This retry policy changes neither sign, time-width, admission,
operator, nor coefficient gates.  Exhaustion at the technical floor is
`BIFURCATION-INCONCLUSIVE`.  Stop refinement only when
the time bracket is at most `dt/64`, the normalized arclength bracket is at
most `1/4096`, and both signs remain resolved.  No checkpoint may be selected
instead after seeing the result.

## Critical mode and coefficients

At the refined critical profile evaluate the physical-normal stability matrix
at 49, 65, and 81 nodes and repeat 81 nodes at twice the Frechet step.  The
principal mode is simple only if it is real, sign-definite, isolated from the
next eigenvalue by at least `1.0`, and the zero is enclosed by the refined
opposite-sign bracket.

For the non-self-adjoint 81-node matrix compute right and left principal
modes, phase them real, and normalize `left^T right = 1`.  Holding the critical
surface fixed, evaluate

The right mode contains the 79 interior physical-normal amplitudes `f`.
Convert it to the full 81-node radial-graph perturbation only through the
same frozen operator construction, `eta=E@(f/w_interior)`, where `E` is the
Neumann extension and `w=f/delta-rho` is the positive physical-normal factor.
Dimension, finiteness, and positivity checks are mandatory.

`a = left^T d(theta_plus)/dt`

with centered dense-trajectory steps `dt/64` and `dt/128`.  Along the physical
normal right mode evaluate

`b = 0.5*left^T d2(theta_plus)/df2`

with centered relative deformation steps `1e-5` and `5e-6`.  Both coefficients
must be finite, nonzero relative to their two-step discrepancy, and preserve
sign under step halving.

## Invariant square-root test

Using the two continuation sides as seeds, solve both branches at the five
prospectively defined offsets

`t=t_* + (1,2,4,8,16)*dt/16`.

At least four offsets must admit both branches.  For one-sided proper cap area,
fit `(A_outer-A_inner)^2` linearly in time.  Require positive slope,
`R^2 >= 0.98`, and a log-fit exponent in `[0.40,0.60]`.  The fit-derived
critical time must agree with the continuation estimate within the larger of
the opposite-sign time span and `dt/64`, the frozen temporal interpolation
resolution.

## Three-grid acceptance

`FREE-BOUNDARY-MOTS-SADDLE-NODE-CLOSURE-PASS` requires every per-grid gate
above, adjacent critical times within `dt/4`, adjacent critical proper
geometry within 1%, consistent signs of `a` and `b`, and adjacent coefficient
transfer within 20% relative or a prespecified numerical floor.  G10 is
executed first; G9 and G11 follow only if G10 passes.

Failure to continue or resolve a coefficient is `BIFURCATION-INCONCLUSIVE`.
A resolved nonzero mode or failed nondegeneracy test is
`SADDLE-NODE-CONDITIONS-NOT-SATISFIED`.  Neither outcome negates Protocol 228.

## Claim boundary

A pass supports a numerical free-boundary saddle-node at finite spacetime and
surface resolution.  It does not prove a continuum theorem, establish an
event horizon or global marginal tube, infer a phase diagram, assign a mass
source, or establish inter-brane transport.
