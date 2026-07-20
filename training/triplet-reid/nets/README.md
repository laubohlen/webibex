Per-file porting provenance for the `tf.contrib.slim` -> `tf_slim` port
(TF1 -> TF2 migration, see `docs/tf1-to-tf2-migration-plan.md` and
`docs/session-notes-2026-07-20-tf2-export-pipeline-verification.md` at the
webibex repo root for the full diagnostic trail and host-side verification
results — this file only summarizes per-file provenance):

- `resnet_utils.py`, `resnet_v1.py`, `resnet_v1_50.py`: copied verbatim from
  `triplet-reid_v2adapted/nets/` — confirmed complete port
  (`tf.contrib.slim` -> `tf_slim`, `tf.variable_scope` ->
  `tf.compat.v1.variable_scope`, etc.). Verified via host-side Phase 2
  (forward-pass smoke test) and Phase 4 (numeric equivalence gate).

- `resnet_v1_101.py`: the "confirmed complete" assumption above was wrong
  for this sibling file — it still called `tf.contrib.slim.arg_scope(...)`
  with no `tf_slim` import. Caught by the AST-based regression test
  `tests/test_export_pipeline.py::test_nets_tf2_has_no_tf_contrib_references`,
  then fixed (see inline `# PORTED` comments in the file).

- `mobilenet_v1.py`, `mobilenet_v1_1_224.py`: the v2adapted reference never
  actually ported these two files (found byte-identical to the TF1
  originals). Ported directly here, using `slim.l2_regularizer`/
  `slim.softmax` (not a literal `tf.keras.regularizers.l2`/`tf.nn.softmax`
  mapping) to preserve the `scope=` kwarg used at the mobilenet call sites.
  Not exercised by the export pipeline (training config uses
  `resnet_v1_50`, not mobilenet) — needs its own Phase 4 verification run
  if it comes into scope later.
