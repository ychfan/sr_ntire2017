import data_tf

def dataset(hr_flist, lr_flist):
    return data_tf.dataset(hr_flist, lr_flist, [2])