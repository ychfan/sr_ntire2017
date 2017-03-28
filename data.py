import tensorflow as tf
import util

resize_func = None

def dataset(hr_flist, lr_flist, scale, resize_func=None):
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
    hr_patches0, lr_patches0 = make_patches(hr_image, lr_image, scale, resize_func)
    hr_patches1, lr_patches1 = make_patches(tf.image.rot90(hr_image), tf.image.rot90(lr_image), scale, resize_func)
    return tf.concat([hr_patches0, hr_patches1], 0), tf.concat([lr_patches0, lr_patches1], 0)

def make_patches(hr_image, lr_image, scale, resize_func):
    hr_image = tf.stack(flip([hr_image]))
    lr_image = tf.stack(flip([lr_image]))    
    hr_patches = util.image_to_patches(hr_image)
    if (resize_func is None):
        lr_patches = util.image_to_patches(lr_image, scale)
    else:
        lr_image = resize_func(lr_image, tf.shape(hr_image)[1:3])
        lr_patches = util.image_to_patches(lr_image)
    return hr_patches, lr_patches 

def flip(img_list):
    flipped_list = []
    for img in img_list:
        flipped_list.append(img)
        flipped_list.append(tf.image.flip_left_right(img))
        flipped_list.append(tf.image.flip_up_down(img))
        flipped_list.append(tf.image.rot90(img, k=2))
    return flipped_list
