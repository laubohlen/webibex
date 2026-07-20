"""Export the triplet-reid resnet_v1_50/fc1024 embedding checkpoint as a TF2 SavedModel.

Collapses `freeze_as_saved_model.py` + `migrate_checkpoint.py` +
`format_saved_model.py` (quarantined at `triplet-reid/_archive/`, see
`_archive/README.md`) into a single export path built on
`tf.saved_model.save()` — NOT `tf.compat.v1.saved_model.builder.SavedModelBuilder`,
which produces no `fingerprint.pb` and was implicated in historical export
bugs (see `ADR-export-pipeline.md`).

HOST EXECUTION: this script requires real TensorFlow 2.18 + tf_slim (not
available in the sandbox that authored it). The Phase 4 verification gate
(R5) has PASSED on host (2026-07-20):
  (a) checkpoint var-name diff: 272/272 exact match,
  (b) np.allclose(atol=1e-4) of the exported embedding against
      test_embedding_old.h5: max abs diff 3.099e-06,
  (c) `saved_model_cli show --all` signature match (input key `bytes_inputs`,
      signature name `serving_default`): confirmed identical.
See docs/tf1-to-tf2-migration-plan.md and
docs/session-notes-2026-07-20-tf2-export-pipeline-verification.md (webibex
repo root) for the full diagnostic trail and runbook commands.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import tensorflow as tf

# nets/heads are real top-level packages once this script lives in the clean
# triplet-reid clone (Phase 1 copies nets_tf2/ -> nets/, heads_tf2/ -> heads/
# alongside this file) -- no import-path shim needed. An earlier sandbox-
# staging version of this file used one (see git history / ADR-export-pipeline.md)
# since nets_tf2/heads_tf2 hadn't been moved into a real package layout yet.
from nets.resnet_v1_50 import endpoints as resnet_v1_50_endpoints
from heads.fc1024 import head as fc1024_head


# --- Confirmed-ground-truth preprocessing contract. Do not change without
# re-verifying against `wibex_model_v03/saved_model.pb` (`strings` output)
# and `triplet-reid/test.ipynb` cell 3 — see ADR-export-pipeline.md. -------
IMAGE_HEIGHT = 288
IMAGE_WIDTH = 144
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)  # (height, width) for tf.image.resize
EMBEDDING_DIM = 128  # experiments/test_inference/args.json: embedding_dim

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = _SCRIPT_DIR / "experiments" / "test_inference" / "checkpoint-4000"
DEFAULT_EXPORT_DIR = _SCRIPT_DIR / "wibex_model_export_staging"

# tf_slim / tf.compat.v1 graph-mode objects (WrappedFunction, Saver, graph-mode
# Variable) have no usable public type stubs in TF 2.18 — `Any` is used
# deliberately here rather than suppressed, per python-types.md's guidance to
# reserve `Any` for genuinely-untyped third-party surfaces.
WrappedForwardFn = Any


def _build_forward_and_restore(
    checkpoint_path: Path,
) -> tuple[WrappedForwardFn, list[tf.Variable]]:
    """Build the resnet_v1_50 + fc1024 forward graph and restore checkpoint weights.

    Uses `tf.compat.v1.wrap_function` — the documented TF2-migration bridge
    for "build a TF1-style (tf_slim) graph, then make it exportable via
    `tf.saved_model.save()`" — rather than freezing weights to constants:
    `wibex_model_v03/variables/` confirms the production SavedModel holds
    real captured variables, not a frozen constant graph, so this export
    must preserve that same shape.

    R4/R7 fix: captures `tf.compat.v1.global_variables()` (NOT
    `trainable_variables()`) after building the forward pass. This is the
    fix for the historical `FailedPreconditionError` on
    `resnet_v1_50/.../BatchNorm/moving_variance` — BatchNorm moving
    mean/variance are non-trainable and were silently left unrestored
    (uninitialized) when a prior version of this pipeline's restore step
    only captured trainable variables.

    Restores via direct eager `.assign()` per variable (see
    `_restore_variables_eagerly`), NOT any session-based mechanism.
    THREE session-based approaches were tried and failed identically
    (`tf.compat.v1.train.Saver.restore()` + implicit read,
    Saver + explicit `tf.convert_to_tensor`, and
    `tf.compat.v1.train.init_from_checkpoint()` +
    `global_variables_initializer()`) — all hit the same
    `capture_by_value` -> `_create_placeholder_helper` failure
    ("You must feed a value for placeholder tensor
    '.../ReadVariableOp.../resource'"). A minimal isolated diagnostic
    (`host_runbook/phase4_debug_wrapfn_probe.py`) confirmed why: variables
    created inside a `wrap_function` trace are genuine live eager
    `ResourceVariable` objects — confirmed consistently across FOUR distinct
    creation patterns (`tf.compat.v1.get_variable`, `tf_slim`'s own conv2d
    weight creation, `tf.Variable` with a tensor `initial_value`, and
    `tf.Variable` with a *callable* `initial_value` — the last being TF's
    own suggested fix for a different, unrelated lifting error, which
    turned out not to matter here). This isn't a bug in how the variables
    are created; it's how `wrap_function` is designed — `WrappedFunction.
    __call__` is the only thing that knows how to feed these captures, and
    a raw `tf.compat.v1.Session.run()` against `wrapped.graph` bypasses
    that entirely, regardless of which variable-creation API produced them.
    Since these are real eager objects, restoring them needs no session at
    all: read each value from the checkpoint by name (Phase 4a already
    confirmed a 272/272 exact name match) and assign it directly. See
    ADR-export-pipeline.md for the three failed session-based attempts.
    """
    captured_vars: list[tf.Variable] = []

    def _forward(images: tf.Tensor) -> tf.Tensor:
        endpoints, _ = resnet_v1_50_endpoints(images, is_training=False)
        endpoints = fc1024_head(endpoints, EMBEDDING_DIM, is_training=False)
        # Called while this FuncGraph is the active default graph (we are
        # inside wrap_function's trace), so this returns exactly the
        # variables belonging to this forward graph.
        captured_vars.extend(tf.compat.v1.global_variables())
        return endpoints["emb"]

    input_spec = tf.TensorSpec(
        shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], dtype=tf.float32, name="images"
    )
    wrapped = tf.compat.v1.wrap_function(_forward, [input_spec])

    if not captured_vars:
        raise RuntimeError("tf.compat.v1.global_variables() captured no variables to restore")

    _restore_variables_eagerly(checkpoint_path, captured_vars)

    return wrapped, captured_vars


def _restore_variables_eagerly(checkpoint_path: Path, variables: list[tf.Variable]) -> None:
    """Load each variable's value directly from the checkpoint via eager
    `.assign()` — no `tf.compat.v1.Session`, no graph execution. See
    `_build_forward_and_restore`'s docstring for why session-based restore
    doesn't work here.
    """
    reader = tf.train.load_checkpoint(str(checkpoint_path))
    for var in variables:
        tensor_name = var.name.split(":")[0]
        var.assign(reader.get_tensor(tensor_name))


class TripletReIDExportModule(tf.Module):
    """`tf.Module` wrapping the restored forward pass + serving preprocessing."""

    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__()
        forward_fn, restored_vars = _build_forward_and_restore(checkpoint_path)
        self._forward = forward_fn
        # Kept as a plain attribute (not just a local) so tf.Module's
        # attribute-tracking picks these up and tf.saved_model.save()
        # serializes them into the SavedModel's variables/ directory.
        self._variables = list(restored_vars)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="bytes_inputs")]
    )
    def serving_default(self, bytes_inputs: tf.Tensor) -> dict[str, tf.Tensor]:
        """Exact preprocessing contract confirmed via `saved_model.pb` strings + test.ipynb.

        NO normalization — `triplet-reid/test.ipynb` has it explicitly
        commented out (RGB-mean subtraction happens separately, inside the
        resnet_v1_50 forward pass itself — see `nets_tf2/resnet_v1_50.py`,
        not here). `decode_jpeg(channels=3)` deliberately also decodes PNG
        (see `triplet-reid/common.py:165-170` comment). Confirmed internal
        op names: `map/while/DecodeJpeg` -> `map/while/resize/ResizeBilinear`,
        wrapped in `tf.map_fn`.
        """

        def _decode_and_resize(image_bytes: tf.Tensor) -> tf.Tensor:
            image_decoded = tf.image.decode_jpeg(image_bytes, channels=3)
            return tf.image.resize(image_decoded, IMAGE_SIZE)

        images = tf.map_fn(_decode_and_resize, bytes_inputs, fn_output_signature=tf.float32)
        embeddings = self._forward(images)
        # Key must be "output_tensor" -- confirmed against wibex_model_v03's
        # actual signature (Phase 3 baseline capture), not "embeddings".
        return {"output_tensor": embeddings}


def export_saved_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    export_dir: Path = DEFAULT_EXPORT_DIR,
) -> Path:
    """Restore `checkpoint_path` and write a TF2 SavedModel to `export_dir`."""
    module = TripletReIDExportModule(checkpoint_path)
    tf.saved_model.save(
        module,
        str(export_dir),
        signatures={"serving_default": module.serving_default},
    )
    return export_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path prefix to the TF1 checkpoint (e.g. .../checkpoint-4000, no extension).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help="Directory to write the TF2 SavedModel to.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    exported_path = export_saved_model(checkpoint_path=args.checkpoint, export_dir=args.export_dir)
    print(f"Exported SavedModel to '{exported_path}'")
