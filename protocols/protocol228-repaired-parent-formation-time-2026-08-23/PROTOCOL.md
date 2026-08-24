# Protocol 228 — Repaired-parent formation-time extension

## Question

Does the parity-repaired, canonical G9/G10/G11 evolution develop the same
donor-brane-capped marginal-surface pair at a later common time, and—if so—do
the pair's proper geometry and principal stability signs close directly across
the three grids?

Protocol 226 established field and physical-tensor spatial closure at
`t/ell = 0.001`.  Protocol 227 then found no admitted donor-capped MOTS on any
of G9/G10/G11 at that time.  The present experiment is the prospectively fixed
later-time successor.  It does not replay or modify the superseded pre-repair
trajectory.

## Immutable inputs

- Exact Protocol-226 authority, result, source, and inherited repaired-parent
  construction/evolution chain.
- Exact Protocol-227 authority, result, detector source, and the three sealed
  states at `t/ell = 0.001`.
- G9 `113 x 211`, G10 `129 x 241`, and G11 `145 x 271` on
  `z=linspace(1,e,Nz)`, `r=linspace(0,10,Nr)`.
- Canonical repaired evolution mode `legacy_wall_axis_outer` and
  `dt/ell = 3.125e-5`.

All bound bytes are authenticated before numerical interpretation.  No parent
solve, projection, state repair, or parameter adjustment is authorized.

## Prospective staged design

The existing common endpoint is global step 32 (`t/ell = 0.001`).  G10 is the
fixed pilot grid.  Evolve it in recoverable 16-step segments and run the exact
Protocol-227 detector at global steps

`48, 64, 80, ..., 256`,

corresponding to `t/ell = 0.0015, 0.0020, ..., 0.0080`.  The detector uses the
unchanged 12 seeds `1.15, 1.20, ..., 1.70`, tolerance `2e-5`, 121 initial
angular nodes, at most 6000 BVP nodes, and 501 dense nodes.  Its residual,
domain, independent-evaluator, and endpoint-clustering gates are exactly those
of Protocol 227.

The candidate time is the earliest prescribed G10 checkpoint containing
exactly two admitted clusters.  No later time may be substituted after seeing
the data.  If no such checkpoint exists through `0.0080`, stop with
`NO-REPAIRED-PARENT-PAIR-THROUGH-0.008`; this is a bounded negative result, not
a nonexistence claim.

If G10 identifies a candidate, evolve G9 and G11 from their exact sealed
`t/ell = 0.001` states to that same global step, again in recoverable 16-step
segments.  Then repeat the full Protocol-227 detector, proper-geometry, and
stability analysis on all three grids.

## Acceptance criteria

`REPAIRED-PARENT-FORMATION-CLOSURE-PASS` requires all of the following at the
preselected common candidate time:

1. exactly two admitted clusters on each of G9/G10/G11;
2. finite positive proper geometry, with outer cap area greater than inner cap
   area on every grid;
3. resolved outward-unstable inner and outward-stable outer principal modes on
   every grid;
4. below 1% adjacent-grid relative transfer for both endpoints and all proper
   geometry observables;
5. adjacent principal eigenvalues agree in classification and differ by less
   than 10% relative or `0.02` absolute;
6. all angular-resolution, Frechet-step, boundary, reality, native-evolution,
   and deterministic artifact checks pass.

A G10 candidate that fails any common-grid condition is classified
`FORMATION-CANDIDATE-NOT-DIRECTLY-CLOSED`.  It cannot be promoted by choosing
a different checkpoint.

## Recovery and claim boundary

Each 16-step evolution endpoint and detector decision is an immutable,
hash-bound artifact.  Recovery adopts only an exact prefix and resumes at the
first absent artifact.  Parents run sequentially; no concurrent factorization
or evolution is used.

A pass establishes only finite-resolution, same-parent later-time formation
and direct geometry/stability closure for the repaired G9/G10/G11 line.  It
does not locate the creation point, prove saddle-node nondegeneracy, establish
a continuous marginal tube or event horizon, infer continuum order, assign a
mass source, or authorize phase-selection, transport, or source-ownership
claims.  Those require separately frozen successors.
