import tensorflow as tf
import util

resize = False
residual = False

def dataset(hr_flist, lr_flist, scale, resize=resize, residual=residual):
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
    if (residual):
        hr_image = make_residual(hr_image, lr_image)
    hr_patches0, lr_patches0 = make_patches(hr_image, lr_image, scale, resize)
    hr_patches1, lr_patches1 = make_patches(tf.image.rot90(hr_image), tf.image.rot90(lr_image), scale, resize)
    lr_patches0 -= 0.5
    lr_patches1 -= 0.5
    return tf.concat([hr_patches0, hr_patches1], 0), tf.concat([lr_patches0, lr_patches1], 0)

def make_residual(hr_image, lr_image):
    hr_image = tf.expand_dims(hr_image, 0)
    lr_image = tf.expand_dims(lr_image, 0)
    hr_image_shape = tf.shape(hr_image)[1:3]
    res_image = hr_image - util.resize_func(lr_image, hr_image_shape)
    return tf.reshape(res_image, [hr_image_shape[0], hr_image_shape[1], 3])

def make_patches(hr_image, lr_image, scale, resize):
    hr_image = tf.stack(flip([hr_image]))
    lr_image = tf.stack(flip([lr_image]))
    hr_image = util.crop_by_pixel(hr_image, 12)
    lr_image = util.crop_by_pixel(lr_image, 12 / scale)
    hr_patches = util.image_to_patches(hr_image)
    if (resize):
        lr_image = util.resize_func(lr_image, tf.shape(hr_image)[1:3])
        lr_patches = util.image_to_patches(lr_image)
    else:
        lr_patches = util.image_to_patches(lr_image, scale)
    return hr_patches, lr_patches 

def flip(img_list):
    flipped_list = []
    for img in img_list:
        flipped_list.append(tf.image.random_flip_up_down(tf.image.random_flip_left_right(img, seed=0), seed=0))
    return flipped_list
