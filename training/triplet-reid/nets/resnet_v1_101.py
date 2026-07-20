import tensorflow as tf
# PORTED (this session): missing `import tf_slim as slim` added — this file
# was copied verbatim from triplet-reid_v2adapted/nets/resnet_v1_101.py,
# which the plan's code-analyst had flagged as "confirmed complete" (unlike
# resnet_v1_50.py, its sibling in the same reference dir). T06's AST-based
# regression test caught that it still called `tf.contrib.slim.arg_scope`
# below — the v2adapted reference was NOT actually complete for this file.
import tf_slim as slim

from nets.resnet_v1 import resnet_v1_101, resnet_arg_scope

_RGB_MEAN = [123.68, 116.78, 103.94]

def endpoints(image, is_training):
    if image.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, 3]')

    image = image - tf.constant(_RGB_MEAN, dtype=tf.float32, shape=(1,1,1,3))

    # PORTED (this session): `tf.contrib.slim.arg_scope` -> `slim.arg_scope`.
    with slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
        _, endpoints = resnet_v1_101(image, num_classes=None, is_training=is_training, global_pool=True)

    endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
        endpoints['resnet_v1_101/block4'], [1, 2], name='pool5')

    return endpoints, 'resnet_v1_101'
