import tensorflow as tf

def build_model(x, training):
    x = tf.layers.conv2d(x, 64, 3, activation=tf.sigmoid, name='conv1')
    x = tf.layers.conv2d(x, 64, 3, activation=tf.sigmoid, name='conv2')
    x = tf.layers.conv2d(x, 64, 3, activation=tf.sigmoid, name='conv3')
    x = tf.layers.conv2d(x, 3, 1, activation=None, name='out')
    return x