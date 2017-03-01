import tensorflow as tf

def build_model(x, reuse):
    hidden_size = 128
    projection_size = 32
    x = conv_gated(x, hidden_size, projection_size, 'conv00', reuse)
    for i in range(10):
        x = crop_by_pixel(x, 1) + conv_gated(x, hidden_size, projection_size, 'conv'+str(i), reuse)
    x = tf.layers.conv2d(x, 1, 1, activation=None, name='out', reuse=reuse)
    return x

def conv_gated(x, hidden_size, projection_size, name, reuse):
    x_filter = tf.layers.conv2d(x, hidden_size, 3, activation=tf.tanh, name=name+'_filt', reuse=reuse)
    x_gate = tf.layers.conv2d(x, hidden_size, 3, activation=tf.sigmoid, name=name+'_gate', reuse=reuse)
    x = x_filter * x_gate
    x = tf.layers.conv2d(x, projection_size, 1, activation=None, name=name+'_proj', reuse=reuse)
    return x

def crop_by_pixel(x, num):
    shape = tf.shape(x)[1:3]
    return tf.slice(x, [0, num, num, 0], [-1, shape[0] - 2 * num, shape[1] - 2 * num, -1])