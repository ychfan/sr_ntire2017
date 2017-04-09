import tensorflow as tf
import util

upsample = True

def build_model(x, scale, training, reuse):
    hidden_size = 128
    bottleneck_size = 64
    x = tf.layers.conv2d(x, hidden_size, 1, activation=None, name='in', reuse=reuse)
    for i in range(6):
        x = util.crop_by_pixel(x, 1) + conv(x, hidden_size, bottleneck_size, training, 'lr_conv'+str(i), reuse)
    x = util.lrelu(x)
    x = tf.image.resize_nearest_neighbor(x, tf.shape(x)[1:3] * scale) + tf.layers.conv2d_transpose(x, hidden_size, scale, strides=scale, activation=None, name='up', reuse=reuse)
    for i in range(4):
        x = util.crop_by_pixel(x, 1) + conv(x, hidden_size, bottleneck_size, training, 'hr_conv'+str(i), reuse)
    x = util.lrelu(x)
    x = tf.layers.conv2d(x, 3, 1, activation=None, name='out', reuse=reuse)
    return x

def conv(x, hidden_size, bottleneck_size, training, name, reuse):
    x = util.lrelu(x)
    x = tf.layers.conv2d(x, bottleneck_size, 1, activation=None, name=name+'_proj', reuse=reuse)

    x = util.lrelu(x)
    x = tf.layers.conv2d(x, hidden_size * 2, 3, activation=None, name=name+'_filt', reuse=reuse)
    x, y = tf.split(x, 2, 3)
    x = x * tf.nn.sigmoid(x)

    return x
