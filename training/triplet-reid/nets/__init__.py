# Used for the commandline flags.
NET_CHOICES = (
    'mobilenet_v1_1_224',
    'resnet_v1_50',
    'resnet_v1_101',
)

# --- tf_slim.batch_norm compatibility patch (TF 2.18 / Keras 3) ---
#
# Root cause (confirmed empirically 2026-07-20, see host_runbook/phase2_debug.sh
# and ADR-export-pipeline.md): tf_slim==1.1.0 (released 2021, predates both
# Keras 3 and the tf_keras companion package) routes batch_norm through
# tensorflow.python.layers.normalization -- a TF1-era compat shim whose
# LazyLoader targets tf_keras.legacy_tf_layers. Confirmed by inspecting the
# installed package directly: tf-keras==2.18.0 does not ship that submodule
# (`has legacy_tf_layers: False`). TF_USE_LEGACY_KERAS=1 does not change this
# -- that env var affects tf.keras/Keras-3 resolution, and this is a
# separate, older tf.layers.* compat path. This is a structural
# incompatibility between tf_slim and TF 2.16+, not a config gap.
#
# Fix: replace tf_slim.batch_norm's implementation. Monkeypatching the
# `slim.batch_norm` attribute in place -- rather than editing every call
# site -- means every existing `normalizer_fn=slim.batch_norm` and
# `arg_scope([slim.batch_norm], ...)` reference across
# resnet_utils.py/resnet_v1.py/mobilenet_v1*.py resolves to this replacement
# automatically, since Python looks up `slim.batch_norm` fresh at each call
# site -- no changes needed in those files.
#
# Must stay @slim.add_arg_scope-decorated: tf_slim's convolution() explicitly
# passes normalizer_params (decay/epsilon/scale/updates_collections, set via
# resnet_arg_scope()'s `normalizer_params=batch_norm_params`) but NOT
# is_training -- that only arrives via resnet_v1.py's separate
# `arg_scope([slim.batch_norm], is_training=...)`, which relies on
# @add_arg_scope's own kwarg auto-injection at call time.
#
# IMPORTANT -- naming, not just behavior, must match the checkpoint:
# a first version of this patch used tf.keras.layers.BatchNormalization,
# which passed Phase 2 (forward pass on random weights) but FAILED Phase 4a's
# checkpoint var-name diff -- every missing variable followed Keras's
# auto-incrementing `batch_normalization_N` naming (e.g.
# `resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/batch_normalization_2/moving_mean`)
# instead of tf_slim's actual convention, a fixed `BatchNorm` sub-scope name
# (the checkpoint was saved by the real tf_slim.batch_norm, which always
# uses that literal name). Building the variables directly with
# tf.compat.v1.get_variable inside variable_scope(scope, "BatchNorm", ...)
# reproduces that naming deterministically -- no dependence on how a Keras
# layer happens to name itself under variable_scope nesting.
#
# Inference-only: this pipeline never trains, so only the is_training=False
# path is implemented (apply the checkpoint's stored moving statistics) --
# no moving-average update ops, no UPDATE_OPS collection wiring.
import tensorflow as tf
import tf_slim as slim


@slim.add_arg_scope
def _batch_norm_compat(
    inputs,
    decay=0.999,
    center=True,
    scale=False,
    epsilon=0.001,
    is_training=True,
    trainable=True,
    scope=None,
    reuse=None,
    **_unused_kwargs,
):
    """Drop-in replacement for tf_slim.layers.batch_norm, checkpoint-name-compatible."""
    with tf.compat.v1.variable_scope(scope, "BatchNorm", [inputs], reuse=reuse):
        params_shape = inputs.shape[-1:]

        beta = None
        if center:
            beta = tf.compat.v1.get_variable(
                "beta",
                params_shape,
                dtype=inputs.dtype,
                initializer=tf.compat.v1.zeros_initializer(),
                trainable=trainable,
            )
        gamma = None
        if scale:
            gamma = tf.compat.v1.get_variable(
                "gamma",
                params_shape,
                dtype=inputs.dtype,
                initializer=tf.compat.v1.ones_initializer(),
                trainable=trainable,
            )
        moving_mean = tf.compat.v1.get_variable(
            "moving_mean",
            params_shape,
            dtype=inputs.dtype,
            initializer=tf.compat.v1.zeros_initializer(),
            trainable=False,
        )
        moving_variance = tf.compat.v1.get_variable(
            "moving_variance",
            params_shape,
            dtype=inputs.dtype,
            initializer=tf.compat.v1.ones_initializer(),
            trainable=False,
        )

        if is_training:
            raise NotImplementedError(
                "This compat batch_norm only supports is_training=False "
                "(inference/export only) -- this pipeline never trains, so "
                "moving-average update ops were deliberately not implemented."
            )

        # Explicit tf.convert_to_tensor before calling batch_normalization --
        # NOT redundant. Confirmed by a real failure: tf.nn.batch_normalization's
        # internal arithmetic operates on its mean/variance/offset/scale
        # arguments directly; passing raw tf.Variable objects (as opposed to
        # plain Tensors) triggered Variable's own operator-overload dispatch
        # (tensorflow/python/ops/variables.py:_run_op) during a
        # tf.compat.v1.wrap_function trace, which got miscaptured as an
        # external placeholder -- restore then crashed with "You must feed a
        # value for placeholder tensor '.../BatchNorm/batchnorm/
        # ReadVariableOp_2/resource'". tf.nn.conv2d, by contrast, explicitly
        # converts its `filters` argument to a tensor internally, which is
        # why the surrounding conv weights never hit this.
        mean_t = tf.convert_to_tensor(moving_mean)
        variance_t = tf.convert_to_tensor(moving_variance)
        beta_t = tf.convert_to_tensor(beta) if beta is not None else None
        gamma_t = tf.convert_to_tensor(gamma) if gamma is not None else None

        return tf.nn.batch_normalization(
            inputs, mean_t, variance_t, beta_t, gamma_t, epsilon
        )


slim.batch_norm = _batch_norm_compat
