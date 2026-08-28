# Protocol 248: three-grid native local-balance transfer

## Question

Do the native-wall-operator local area and quasi-local flux-balance diagnostics admitted on G10
transfer to the authenticated G9/G10/G11 outer marginal tube at matched late-time leaves?

Protocol 247 prospectively authorized this archive-only test.  No spacetime evolution, surface
solve, parent construction, or submitted-paper edit is permitted.

Execution is restricted to the same numerical ABI used by Protocol 245: Linux AArch64,
Python 3.8.10, NumPy 1.24.4, and SciPy 1.10.1, with all five numerical thread controls fixed
to one before imports.

## Frozen data and evaluation

The calculation uses Protocol 247 G9/G11 states and outer profiles at steps 43--48, the Protocol
245 G10 full-timestep balance record, the Protocol 246 temporal comparison, and the Protocol 245
physical evaluator with only its hard-coded G10 array dimensions and output grid label generalized
to the authenticated input grid.  Its equations, stencils, paths, and local gates are unchanged.
Steps 44--47 are classification-driving because each centered rate uses its two adjacent leaves.
Stencil widths 5, 7, and 9 are all required.  Every G9/G11 evaluation and the complete adjacent-grid
comparison are repeated exactly.

The comparison uses the Protocol 246 ceilings without alteration: 1% for area values, seam
geometry, target rate, total flux, and ledger-term errors normalized by the balance scale; 2%
for the two area-rate constructions; a 1% local normalized-ledger residual; and the inherited
5% relative or 0.005 absolute rule for native directional, history, and wall rates.

## Ordered gates

1. Protocols 245--247 are authentic passing prerequisites.
2. G9, G10, and G11 each pass all inherited local orientation, area, seam, native-wall-rate,
   and flux-ledger gates.
3. Matched times agree exactly.
4. Area, charge/flux, and native-rate signs agree.
5. Geometry and area-rate transfers pass.
6. Flux-ledger and individual-term transfers pass.
7. Native endpoint-operator rates pass.

## Meaning

A pass establishes bounded three-grid spatial transfer of the native local balance diagnostics
on four late outer-tube leaves, together with the independent half-timestep consistency already
established by Protocol 246.  It does not establish a continuum theorem, integrated/global
balance, event horizon, connected bulk topology, mass transfer, or source ownership.
