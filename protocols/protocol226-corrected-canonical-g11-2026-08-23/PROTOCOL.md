# Protocol 226 — Corrected recoverable canonical G11 spatial closure

## Question

Does the complete-chain G11 parent resolve the shallow auxiliary `q`/`v` RMS trend
while preserving the already established physical-tensor spatial improvement
at `t=0.001`?

## Correction to Protocol 225

Protocol 224 constructed the correct complete-chain G11 parent, but its stored
zero-step acceleration was evaluated with the second-mode
`wall_owner_last_experimental` policy. Protocol 225 evolved with the canonical
`legacy_wall_axis_outer` policy and incorrectly required its first acceleration
to equal that second-mode array. Its first trajectory therefore completed with
every native gate passing but was discarded by an impossible cross-mode replay
check. Protocol 225 is retained as a technical negative and makes no scientific
statement about G11.

Protocol 226 prospectively distinguishes the two policies. Before each
trajectory it evaluates the canonical zero-step acceleration twice, requires
bitwise identity and all native gates, requires the first RK stage to reproduce
that canonical value, and separately verifies that the Protocol-224 archived
array reproduces the second-mode value and differs from the canonical value.

## Frozen test

- Use the Protocol-224 complete-chain G11 parent (145 by 271).
- Use the unchanged canonical `legacy_wall_axis_outer` evolution, binary64 RK2,
  `dt=0.00003125`, exactly 32 steps, and endpoint time `0.001`.
- Execute G11 twice.  Both full 64-stage traces, endpoint states, and endpoint
  accelerations must be bitwise identical.
- Every inherited stage and endpoint technical gate must pass.
- Compare G10 and G11 endpoint increments on their exact 17-by-31 common
  lattice.  For `q`, `v`, source, and memory, both maximum and RMS differences
  must be smaller than the sealed Protocol-220 G9–G10 values, except for the
  inherited 64-ulp roundoff rule.
- The proper-volume differences in the physical metric increment and ADM
  extrinsic-curvature tensor must be smaller than the Protocol-220 G9–G10
  values, and their G10–G11 relative differences must remain below `0.002`.

If the physical tensors pass but `q` or `v` RMS does not decrease, the result is
classified as an auxiliary RMS plateau rather than a physical-tensor failure.
That outcome closes the experiment with an explicit limitation; it does not
authorize further threshold changes.  Failure of a physical-tensor gate blocks
the paper-facing spatial closure.

No outcome establishes a continuum order, authorizes Phase A or production
evolution, or supports an evolved-physics claim.

## Recovery

The two G11 trajectories are immutable sequential checkpoints.  Exact prefixes
are adopted after restart; a lone archive publication gap is rerun and must
match before its receipt is published.  A two-hour watchdog may restart a
stopped process against this recovery entry point.
