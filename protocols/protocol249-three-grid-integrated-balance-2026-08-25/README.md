# Protocol 249

Archive-only G9/G10/G11 finite-segment integration of the native quasi-local charge/flux balance.

- `PROTOCOL.md`: prospective segment, quadrature, gates, thresholds, and claim boundary.
- `runner.py`: authenticates prior records, reevaluates charges, integrates the ledger twice, and
  publishes only new Protocol 249 artifacts.
- `engine248.py` and `src/`: inherited physical charge evaluator and exact source closure.
- `integrated_balance_core.py`: pure quadrature, comparison, and ordered-classification logic.
- `sealed-inputs/`: immutable Protocol 244/246/247/248 records and historical configuration.
- `candidate-output/`: absent at freeze; the only permitted result namespace.

The submitted PRD paper is outside this capsule and is not edited.
