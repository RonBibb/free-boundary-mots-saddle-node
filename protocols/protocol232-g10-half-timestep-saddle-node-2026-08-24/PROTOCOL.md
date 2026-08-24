# Protocol 232 — G10 half-timestep saddle-node closure

## Question

Does the Protocol-229 G10 free-boundary saddle-node persist when the repaired
parent is independently re-evolved from `t/ell=0.001` to `0.0015` with the
same explicit-midpoint RK2 method at half the timestep?

This is the prespecified temporal discriminator requested during manuscript
review.  It changes no equation, parent, spatial grid, surface class,
stability operator, continuation rule, physical parameter, or scientific
threshold other than replacing the parent timestep and its derived temporal
resolution scales by half their Protocol-229 values.

## Immutable inputs

- Protocol 229 v4 and Protocol 230 exact authority/result chain.
- Protocol 231 existing-data precision audit.
- Exact Protocol-228 G10 repaired state at `t/ell=0.001`, its two accepted
  G10 profiles at `t/ell=0.0015`, and the complete inherited source/runtime
  chain authenticated transitively by Protocol 229.

The coarse result must be
`FREE-BOUNDARY-MOTS-SADDLE-NODE-CLOSURE-PASS`.  No coarse artifact is modified.

## Half-timestep parent

Keep G10 fixed at `129 x 241` over the same domain.  Evolve the exact repaired
state from `t/ell=0.001` to `0.0015` using

`dt_half/ell = 1.5625e-5`,

32 explicit-midpoint RK2 steps.  Archive immutable checkpoints every eight
steps.  All inherited finite, Lorentzian, wall/axis, source, and endpoint
technical gates must pass.  Refine the two Protocol-228 terminal profiles on
the half-step endpoint using the unchanged local BVP and require the inherited
admission gates and opposite principal stability signs.

## Continuation and critical calculation

Repeat the Protocol-229 G10 continuation using the half-step RK2 dense
extension.  All surface, pseudo-arclength, bracket, operator, coefficient,
and normal-form rules are inherited unchanged.  Timestep-derived quantities
use `dt_half`: the zero bracket is at most `dt_half/64`; the temporal
coefficient differences use `dt_half/64` and `dt_half/128`; and the five
square-root offsets are `(1,2,4,8,16) dt_half/16`.

## Prospective interpretation

`G10-HALF-TIMESTEP-SADDLE-NODE-CLOSURE-PASS` requires:

1. every inherited half-step G10 saddle-node gate passes;
2. the critical-time shift is at most the original Protocol-229 `dt/64`;
3. critical proper area differs by less than 1%;
4. transversality and quadratic coefficients preserve sign and differ by less
   than 20% relative;
5. both coarse and half-step exponents remain in the inherited `[0.40,0.60]`
   interval.

These are inherited Protocol-229 transfer scales, not post-outcome fits.  A
passing result preserves the finite-resolution saddle-node classification and
calibrates reportable temporal precision.  A passing structure with a failed
transfer criterion is
`MATERIAL-TEMPORAL-DRIFT-REQUIRES-DT-QUARTER`; it does not negate the fold.
Failure to resolve the half-step continuation is
`HALF-TIMESTEP-BIFURCATION-INCONCLUSIVE`.  A resolved failure of a fold gate is
`HALF-TIMESTEP-SADDLE-NODE-CONDITIONS-NOT-SATISFIED`.

No continuum theorem, full nonsymmetric spectral claim, event horizon, phase
selection, transport, or source-ownership inference is authorized.
