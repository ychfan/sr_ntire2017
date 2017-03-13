import tensorflow as tf
import data_naive

def dataset(hr_flist, lr_flist):
    return data_naive.dataset(hr_flist, lr_flist, tf.image.resize_nearest_neighbor)