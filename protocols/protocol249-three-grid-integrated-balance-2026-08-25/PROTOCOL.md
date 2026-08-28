# Protocol 249: three-grid finite-segment integrated balance

## Question

Does the native local quasi-local balance admitted by Protocol 248 close after integration over
a fixed finite segment of the authenticated G9/G10/G11 outer marginal tube?

This is an archive-only successor authorized by Protocol 248. It performs no spacetime evolution,
surface solve, parent construction, or submitted-paper edit.

## Frozen segment and observable

The classification-driving segment is fixed before evaluation to steps 44--47,
`0.001375 <= t/ell <= 0.00146875`. The quasi-local charge is independently reevaluated at all four
nodes for stencil widths 5, 7, and 9. Rates and every individual bulk/brane ledger term are taken
from the authenticated centered-path Protocol 248 records. Integration uses the four-node composite
trapezoid rule, fixed time step `3.125e-5`, and no fitted parameter.

The comparison distinguishes two statements:

1. charge change versus the integrated finite-difference target rate tests time quadrature;
2. charge change versus the integrated physical flux ledger tests the finite-segment balance.

The ledger must contain the brane-endpoint terms `coupled_seam_global_radius` and
`coupled_seam_joint_work`. Their integrals and every other term are archived separately.

## Frozen ceilings

The inherited Protocol 246/248 ceiling of strict relative error `< 0.01` applies to charge-target
quadrature closure, charge-flux closure, target-flux closure, stencil comparisons, adjacent-grid
charge values and changes, integrated target/total fluxes, the brane-endpoint subtotal, and each
integrated ledger term normalized by the larger integrated balance norm. The term sum must reproduce
the integrated total flux to `< 1e-12` of the integrated balance norm.

## Ordered gates

1. Protocols 244, 246, 247, and 248 are authentic passing prerequisites.
2. Charge change, integrated target rate, and integrated total flux are positive.
3. Charge change agrees with the integrated target rate.
4. Charge change agrees with the integrated physical flux ledger.
5. The exact brane-inclusive term inventory sums to the integrated total flux.
6. Widths 5, 7, and 9 agree within each grid.
7. G9--G10 and G10--G11 transfer passes at every width.
8. Protocol 246's independent G10 half-step local-balance control remains admitted.

Every charge evaluation and the complete integration/comparison are repeated exactly.

## Runtime and meaning

Execution is restricted to Linux AArch64, Python 3.8.10, NumPy 1.24.4, and SciPy 1.10.1 with all
five numerical-library thread controls fixed to one before imports.

A pass establishes a balanced finite outer marginal-tube segment at finite resolution, with bounded
three-grid spatial transfer and the separately established local half-step control. It does not
establish a continuum theorem, an event horizon, connected bulk topology, a globally conserved
inter-sector charge, mass transfer, or source ownership.
