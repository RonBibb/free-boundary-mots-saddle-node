# Protocol 231 — Existing-data saddle-node precision audit

## Purpose

Before any new evolution, audit the immutable Protocol-229-v4 G9/G10/G11
records for the numerical-precision issues raised during manuscript review.
This is a derived, read-only analysis.  It does not continue a surface,
evaluate an equation, modify a parent, or change any Protocol-229 result.

## Immutable inputs

- The exact Protocol-229-v4 authority, implementation, continuation core, and
  three grid JSON/NPZ records.
- The exact Protocol-230 archive-only finalization result.

## Prespecified diagnostics

For each grid, use the archived five `(time, area separation)` pairs and the
independent continuation midpoint `t_*` to report:

1. the free log--log exponent and its log-fit `R^2` with `t_*` held fixed;
2. the maximum relative residual of the fixed exponent `p=1/2` amplitude fit;
3. leave-one-out free-exponent sensitivity with `t_*` held fixed;
4. drop-nearest and drop-farthest sensitivity;
5. the archived coupled fit and the distinction between its squared-area
   linear-fit `R^2` and its separately computed log exponent.

For `t_*`, critical area, transversality, and quadratic coefficient, report
the three-grid sequence, absolute and relative spread, monotonicity, and
whether a real Richardson order is defined.  A sign-reversing adjacent
difference is reported as non-asymptotic scatter; no order is invented.

Inspect the bound implementation to establish the exact mode convention:
the right mode is phased by its largest-magnitude component, normalized to
unit infinity norm, and the left Euclidean matrix-adjoint mode is then scaled
to unit biorthogonal overlap.  The post-normalization overlap is therefore a
construction identity, not an independent diagnostic.

Inspect the bound detector/continuation sources to distinguish the primary
spectral detector's `10^5` condition-number rule from Protocol 229's
load-bearing local-BVP continuation.  This audit may establish whether the
condition gate is load-bearing; it does not manufacture a condition-number
trace that was never archived.

## Output and boundary

The sole result is `candidate-output/protocol231_result.json`, classified
`EXISTING-DATA-PRECISION-AUDIT-COMPLETE`.  It may motivate new temporal work
or manuscript corrections, but cannot reclassify Protocol 229, establish a
continuum order, or authorize any additional physical claim.
