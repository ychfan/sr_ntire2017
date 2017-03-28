import tensorflow as tf
import data

resize_func = tf.image.resize_nearest_neighbor

def dataset(hr_flist, lr_flist, scale):
    return data.dataset(hr_flist, lr_flist, scale, resize_func)