# Verification image: TensorFlow 2.18.0 (original, pre-2.18.1-patch baseline).
# Reproduces the exact environment the export pipeline was first verified
# against (see docs/tf1-to-tf2-migration-plan.md). Built once, layer-cached;
# every actual execution step should run with --network=none.
#
# Installed normally (not --target/PYTHONPATH) so tf_keras's internal
# dynamic-import machinery resolves correctly.
FROM tensorflow/tensorflow:2.18.0

# tf_slim: nets/*.py use tf.contrib.slim-equivalent APIs via this package
#   (see docs/tf1-to-tf2-migration-plan.md).
# tf-keras: TF 2.16+ defaults to Keras 3, but tf_slim.batch_norm still calls
#   into the legacy Keras 2 BatchNormalization layer, which needs this.
# h5py: reads test_embedding_old.h5 in the numeric verification gate.
#   Deliberately unpinned -- the tensorflow/tensorflow:2.18.0 base image
#   already bundles a compatible h5py; --no-deps makes this a no-op. A
#   floor pin of h5py>=3.14.0 was tried here but reverted (2026-07-23):
#   only confirmed against the 2.21.0 base (tf2210), not this 2.18.0 one,
#   and this image's own gate-passed result must not be risked on an
#   unverified version bump. Pin only after re-running verify_gate.sh here.
RUN pip install --no-cache-dir --no-deps --only-binary=:all: \
    tf_slim==1.1.0 \
    tf-keras==2.18.0 \
    h5py
