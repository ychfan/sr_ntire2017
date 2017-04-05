import tensorflow as tf
import util

upsample = True


def build_model(x, scale, training, reuse):
    hidden_size = 128
    bottleneck_size = 32
    survival_rate = 0.5
    survival_rate = tf.constant(survival_rate, name='survival_rate')

    x = tf.layers.conv2d(
        x, hidden_size, 1, activation=None, name='in', reuse=reuse)
    for i in range(5):
        x = util.crop_by_pixel(
            x, 1) + conv(x, hidden_size, bottleneck_size, training, 'lr_conv' +
                         str(i), reuse)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d_transpose(
        x,
        hidden_size,
        scale,
        strides=scale,
        activation=None,
        name='up',
        reuse=reuse)
    print x.get_shape().as_list()
    for i in range(5):
        shortcut = util.crop_by_pixel(x, 1)
        #print shortcut.get_shape().as_list()
        resblock = conv(
            x,
            hidden_size,
            bottleneck_size,
            training,
            'hr_conv_share',
            reuse=None if i == 0 else True)
        if training:
            survival_roll = tf.random_uniform(
                shape=[], minval=0.0, maxval=1.0, name='suvival' + str(i))
            survive = tf.less(survival_roll, survival_rate)
            dummy_zero = tf.zeros_like(resblock)
            x = tf.cond(survive, lambda: tf.add(shortcut, resblock),
                        lambda: tf.add(dummy_zero, shortcut))
        else:
            x = tf.add(tf.mul(resblock, survival_rate), shortcut)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, 3, 1, activation=None, name='out', reuse=reuse)
    return x


def conv(x, hidden_size, bottleneck_size, training, name, reuse):
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(
        x,
        bottleneck_size,
        1,
        activation=None,
        name=name + '_proj',
        reuse=reuse)

    x = tf.nn.relu(x)
    x = tf.layers.conv2d(
        x,
        bottleneck_size,
        3,
        activation=None,
        name=name + '_filt',
        reuse=reuse)

    x = tf.nn.relu(x)
    x = tf.layers.conv2d(
        x, hidden_size, 1, activation=None, name=name + '_recv', reuse=reuse)
    return x
