# Verification image: TensorFlow 2.18.1 (patch bump — see
# docs/tf1-to-tf2-migration-plan.md "TF 2.18.0 -> 2.18.1 patch bump").
#
# No matching tensorflow/tensorflow:2.18.1 Docker Hub tag exists (confirmed:
# `docker manifest inspect tensorflow/tensorflow:2.18.1` 404s, only 2.18.0
# was ever published as an image for this patch version) -- so this stays
# FROM the 2.18.0 base image and upgrades tensorflow via pip on top of it.
FROM tensorflow/tensorflow:2.18.0

# Upgrades tensorflow on top of the 2.18.0 base image. Fixes 8 curl CVEs
# (see TF 2.18.1 release notes); no API changes affecting this pipeline
# (tf.lite/TPU-only breaking changes, neither used here).
RUN pip install --no-cache-dir --only-binary=:all: tensorflow==2.18.1

# tf_slim: nets/*.py use tf.contrib.slim-equivalent APIs via this package
#   (see docs/tf1-to-tf2-migration-plan.md).
# tf-keras: stays pinned at 2.18.0 -- tf-keras never published a 2.18.1
#   (confirmed via PyPI: only one 2.18.x release exists), it tracks TF's
#   minor version, not patch.
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
