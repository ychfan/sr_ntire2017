import tensorflow as tf
import data

def dataset(hr_flist, lr_flist, resize_func):
    with open(hr_flist) as f:
        hr_filename_list = f.read().splitlines()
    with open(lr_flist) as f:
        lr_filename_list = f.read().splitlines()
    filename_queue = tf.train.slice_input_producer([hr_filename_list, lr_filename_list], num_epochs=1)
    hr_image_file = tf.read_file(filename_queue[0])
    lr_image_file = tf.read_file(filename_queue[1])
    hr_image = tf.image.decode_image(hr_image_file, channels=3)
    lr_image = tf.image.decode_image(lr_image_file, channels=3)
    hr_image = tf.image.convert_image_dtype(hr_image, tf.float32)
    lr_image = tf.image.convert_image_dtype(lr_image, tf.float32)
    hr_image = tf.stack([hr_image, tf.image.flip_left_right(hr_image)])
    lr_image = tf.stack([lr_image, tf.image.flip_left_right(lr_image)])
    lr_image = resize_func(lr_image, tf.shape(hr_image)[1:3])
    hr_patches = data.image_to_patches(hr_image)
    lr_patches = data.image_to_patches(lr_image)
    return hr_patches, lr_patches