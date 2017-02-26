import tensorflow as tf

def build_model(x):
    hidden_size = 128
    projection_size = 32
    for i in range(10):
        reuse = False
        if (i > 0):
            reuse = True
        x = res_group(x, hidden_size, projection_size, 'group', reuse)
    return x

def res_group(x, hidden_size, projection_size, name, reuse):
    prev_x = x
    x_filter = tf.layers.conv2d(x, hidden_size, 3, padding='same', activation=tf.tanh, name=name+'_filt', reuse=reuse)
    x_gate = tf.layers.conv2d(x, hidden_size, 3, padding='same', activation=tf.sigmoid, name=name+'_gate', reuse=reuse)
    x = x_filter * x_gate
    x = tf.layers.conv2d(x, projection_size, 1, activation=None, name=name+'_proj', reuse=reuse)
    for i in range(6):
        x = conv_res(x, hidden_size, projection_size, name+'_res'+str(i), reuse)
    x = tf.layers.conv2d(x, 1, 1, activation=None, name=name+'_out', reuse=reuse)
    x += prev_x
    return x
    

def conv_res(x, hidden_size, projection_size, name, reuse):
    prev_x = x
    x_filter = tf.layers.conv2d(x, hidden_size, 3, padding='same', activation=tf.tanh, name=name+'_filt', reuse=reuse)
    x_gate = tf.layers.conv2d(x, hidden_size, 3, padding='same', activation=tf.sigmoid, name=name+'_gate', reuse=reuse)
    x = x_filter * x_gate
    x = tf.layers.conv2d(x, projection_size, 1, activation=None, name=name+'_proj', reuse=reuse)
    x += prev_x
    return x
