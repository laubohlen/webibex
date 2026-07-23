# Verification image: TensorFlow 2.21.0 -- see
# docs/tf1-to-tf2-migration-plan.md "TF 2.21.0 migration".
#
# Unlike 2.18.1, a matching tensorflow/tensorflow:2.21.0 Docker Hub tag
# DOES exist (confirmed via Docker Hub before starting this migration), so
# this builds FROM it directly -- no pip-upgrade-on-top-of-base-image
# workaround needed here.
FROM tensorflow/tensorflow:2.21.0

# tf_slim: nets/*.py use tf.contrib.slim-equivalent APIs via this package
#   (see docs/tf1-to-tf2-migration-plan.md). Version unchanged from 2.18.x --
#   if this build fails to resolve/import, that's the first thing to check.
# tf-keras: pinned to 2.21.0, matching TF's minor version (tf-keras tracks
#   TF's minor, not patch -- confirmed via PyPI, same pattern as the 2.18.x
#   line, which only ever had one 2.18.0 tf-keras release).
# h5py: reads test_embedding_old.h5 in the numeric verification gate.
#   Floor pin (not exact) -- the tensorflow/tensorflow base image already
#   bundles a compatible h5py; --no-deps means this is a no-op when already
#   satisfied. 3.14.0 confirmed present in this base image this session
#   (2026-07-23), hence the floor.
RUN pip install --no-cache-dir --no-deps --only-binary=:all: \
    tf_slim==1.1.0 \
    tf-keras==2.21.0 \
    "h5py>=3.14.0"
