import tensorflow as tf

upsample = False

def build_model(x, scale, training, reuse):
    x = tf.layers.conv2d(x, 64, 3, activation=tf.sigmoid, name='conv1', reuse=reuse)
    x = tf.layers.conv2d(x, 64, 3, activation=tf.sigmoid, name='conv2', reuse=reuse)
    x = tf.layers.conv2d(x, 64, 3, activation=tf.sigmoid, name='conv3', reuse=reuse)
    x = tf.layers.conv2d(x, 3, 1, activation=None, name='out', reuse=reuse)
    return x