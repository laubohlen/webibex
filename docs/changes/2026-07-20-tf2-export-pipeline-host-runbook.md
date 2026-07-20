# CR: TF1→TF2 triplet-reid export pipeline — host verification runbook + code fixes

**What changed:**
- New: `webibex/tmp/inference/host_runbook/` (gitignored scratch, host-only tooling) — Docker-based, `--network=none`-by-default verification tooling for the export pipeline (`Dockerfile.pipdeps`, `lib/common.sh`, 5 `phaseN_*.sh` orchestration scripts, plus Python probes/diagnostics for each phase, including a debug/repro toolkit for the wrap_function eager-variable issue).
- Code fix, now committed at `webibex/training/triplet-reid/nets/__init__.py`: `_batch_norm_compat` monkeypatches `slim.batch_norm` for TF 2.16+/Keras 3 compatibility, with checkpoint-matching `BatchNorm` variable naming.
- Code fix, now committed at `webibex/training/triplet-reid/export_saved_model.py`: restore mechanism switched from session-based (`Saver`/`init_from_checkpoint`) to eager `.assign()` per variable; output signature key fixed to `output_tensor`; import shim removed (`nets`/`heads` are real packages at their committed location).
- Doc: `webibex/tmp/inference/ADR-export-pipeline.md` (gitignored scratch) updated with the full diagnostic trail for both the `batch_norm` and restore-mechanism fixes — durable copy folded into `docs/tf1-to-tf2-migration-plan.md`.

**Follow-up action — DONE (2026-07-20):** Phase 5 (R6) complete. Consolidated into `webibex/training/triplet-reid/` (committed, tracked by webibex's own git — not the nested `triplet-reid-clean/` clone under `tmp/inference/`, which had no durable remote). The five-directory sprawl (`wibex_model_v01/v02`, `model/`, `new_model2/`, `saved_model/`) under `tmp/inference/` is still pending removal (`wibex_model_v03/` correctly retained as the R7 baseline) — separate follow-up, not blocking.

**Do NOT:**
- Delete `wibex_model_v03/` during any cleanup — it's the R7 numeric baseline, still referenced by the verification gate.
- Re-derive the `batch_norm` or restore-mechanism fixes from scratch in a future session — both required multiple failed attempts and a purpose-built isolated diagnostic to root-cause; read `ADR-export-pipeline.md` first.
- Re-run `host_runbook/phase1_clean_clone.sh` against an existing `triplet-reid-clean/` without removing it first — the script refuses on purpose (would silently clobber in-progress work), but don't work around that by force-deleting without checking its contents first.

**Trigger:** whenever picking up Phase 5, or re-running any `host_runbook/` script on a fresh clone of this state.

**Why:** the prior export pipeline was 3 undocumented manual scripts, implicated in historical export bugs (a `FailedPreconditionError` from an incomplete variable-capture bug, and numeric drift from an old `SavedModelBuilder`-based export path). This rebuild produces one tested, `tf.saved_model.save()`-based export script with a mandatory, automated verification gate (checkpoint name-diff, numeric equivalence, signature parity) instead of trusting a manual process.

**Verify:** `bash host_runbook/phase4_verification_gate.sh` — all three parts (4a/4b/4c) must pass. Confirmed passing as of 2026-07-20 (log: `max abs diff: 3.099e-06` at `atol=1e-4`; 272/272 checkpoint variable names matched; `serving_default` signature identical to production's captured baseline).

**Rollback:** `webibex/training/triplet-reid/` is now committed to webibex's git history (this CR doc's own commit). To roll back the code, `git revert`/remove that commit — it doesn't touch production (RunPod loads from a separate `embedding_model` dir, confirmed unrelated). The host-only tooling and diagnostics (`tmp/inference/host_runbook/`, `ADR-export-pipeline.md`, `triplet-reid-clean/`, `wibex_export_verified/`) remain gitignored scratch space, untouched by any rollback of the committed code.
