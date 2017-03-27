import tensorflow as tf

def build_model(x, training):
    hidden_size = 128
    bottleneck_size = 32
    x = tf.layers.conv2d(x, hidden_size, 9, activation=None, name='in')
    for i in range(10):
        x = crop_by_pixel(x, 1) + conv(x, hidden_size, bottleneck_size, training, 'conv'+str(i), False)
    x = tf.layers.conv2d(x, 3, 1, activation=None, name='out')
    return x

def conv(x, hidden_size, bottleneck_size, training, name, reuse):
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, bottleneck_size, 1, activation=None, name=name+'_proj', reuse=reuse)
    
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, bottleneck_size, 3, activation=None, name=name+'_filt', reuse=reuse)
    
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, hidden_size, 1, activation=None, name=name+'_recv', reuse=reuse)
    return x

def crop_by_pixel(x, num):
    shape = tf.shape(x)[1:3]
    return tf.slice(x, [0, num, num, 0], [-1, shape[0] - 2 * num, shape[1] - 2 * num, -1])
