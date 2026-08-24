# Protocol 230: archive-only Protocol 229 finalization

Status: prospective until the source and input authority is generated; immutable thereafter.

## Question

Can the three completed, immutable Protocol 229 v4 grid records be authenticated and
combined under the already-frozen Protocol 229 adjacent-grid gates after its final JSON
publication failed solely because a NumPy Boolean was not converted to a JSON-native
Boolean?

## Scope

This protocol performs no evolution, continuation, eigensolve, root solve, interpolation,
operator application, or scientific reevaluation. It does not alter Protocol 229 or any of
its six grid artifacts. It authenticates the Protocol 229 authority and complete bound
source/input chain, authenticates each grid JSON fingerprint and its NPZ file record, requires
every frozen per-grid gate to have passed, and evaluates only the two already-prospective
adjacent-grid comparisons.

The output is a finalization record, not a new scientific experiment. Every Boolean written
by this protocol is an exact built-in JSON Boolean.

## Immutable inputs

The freeze authority binds:

- the Protocol 229 v4 freeze authority and source;
- the G9, G10, and G11 grid JSON records;
- the three corresponding NPZ archives.

The Protocol 229 authority must independently reproduce its own fingerprint, and all source
and input file records nested in it must still match. Each grid record must reproduce the
Protocol 229 grid fingerprint, carry the exact Protocol 229 authority SHA-256, bind the exact
NPZ bytes, and report all nine per-grid gates as built-in `true`.

## Frozen adjacent-grid gates

For G9--G10 and G10--G11, require all of the following, exactly as in Protocol 229 v4:

- absolute critical-time difference no greater than `dt/4`, with `dt=0.00003125`;
- relative critical-area difference less than `0.01`;
- matching nonzero transversality signs and relative difference less than `0.20`;
- matching nonzero quadratic-coefficient signs and relative difference less than `0.20`.

No threshold may be changed after reading the artifacts.

## Classification

The sole PASS classification is
`FREE-BOUNDARY-MOTS-SADDLE-NODE-CLOSURE-PASS`. It requires all three authenticated per-grid
records and both adjacent-grid gates to pass. Otherwise the classification is
`SADDLE-NODE-CONDITIONS-NOT-SATISFIED` or the process fails closed on an integrity error.

A PASS authorizes the bounded statement that the archived calculation numerically resolves a
free-boundary MOTS saddle-node across G9, G10, and G11 under the frozen finite-resolution
criteria. It does not authorize a continuum theorem, event-horizon claim, phase-selection
claim, or source-ownership claim.

## Output

The only candidate artifact is `candidate-output/protocol230_result.json`. It records the
complete input file records, the authenticated grid summaries, the adjacent-grid metrics and
gates, the bounded claim firewall, and a canonical SHA-256 fingerprint. Existing output is
adopted only after full validation; partial or different output fails closed.
