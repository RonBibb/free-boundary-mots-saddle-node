# Protocol 248

Archive-only G9/G10/G11 spatial transfer of the native local area/flux-balance diagnostics.

- `PROTOCOL.md`: frozen question, data, gates, and claim boundary.
- `runner.py`: authenticates Protocols 245--247, evaluates G9/G11 twice, compares adjacent grids, and publishes new results.
- `engine245.py`: Protocol 245 evaluator, generalized only to derive the grid shape and record the selected label.
- `dense_balance_core.py`: inherited local-balance helpers.
- `spatial_balance_core.py`: pure Protocol 246-derived spatial comparisons and ordered classification.
- `src/`: exact inherited balance source closure.
- `sealed-inputs/`: copied immutable parents; no prior data are modified.
- `tests/`: manufactured comparison and taxonomy checks.
- `candidate-output/`: new Protocol 248 result namespace, absent at freeze.

This capsule performs no evolution or surface solve and does not edit the submitted paper.
