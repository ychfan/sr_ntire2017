import tensorflow as tf
import data

resize = False
residual = True

def dataset(hr_flist, lr_flist, scale):
    return data.dataset(hr_flist, lr_flist, scale, resize, residual)