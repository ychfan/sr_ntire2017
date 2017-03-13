import tensorflow as tf
import data

def dataset(hr_flist, lr_flist, scale_list):
    distort = True
    with open(hr_flist) as f:
        hr_filename_list = f.read().splitlines()
    filename_queue = tf.train.string_input_producer(hr_filename_list, num_epochs=1)
    reader = tf.WholeFileReader()
    _, image_file = reader.read(filename_queue)
    hr_image = tf.image.decode_image(image_file, channels=3)
    hr_image = tf.image.convert_image_dtype(hr_image, tf.float32)
    if distort:
        hr_image = distort_image(hr_image)
    else:
        hr_image = tf.expand_dims(hr_image, 0)
    hr_patches = data.image_to_patches(hr_image)
    hr_patches_list = [hr_patches] * len(scale_list)
    lr_image_list = []
    for scale in scale_list:
        lr_image = tf.image.resize_bicubic(tf.image.resize_bicubic(hr_image, tf.shape(hr_image)[1:3] / scale), tf.shape(hr_image)[1:3])
        lr_image_list.append(lr_image)
    lr_patches = data.image_to_patches(tf.concat(lr_image_list, 0))
    return tf.concat(hr_patches_list, 0), lr_patches

def distort_image(image):
    image1 = tf.image.random_flip_left_right(image)
    image1 = tf.image.random_brightness(image1, max_delta=32. / 255.)
    image1 = tf.image.random_saturation(image1, lower=0.5, upper=1.5)
    image1 = tf.image.random_hue(image1, max_delta=0.2)
    image1 = tf.image.random_contrast(image1, lower=0.5, upper=1.5)
    image2 = tf.image.random_flip_left_right(image)
    image2 = tf.image.random_brightness(image2, max_delta=32. / 255.)
    image2 = tf.image.random_contrast(image2, lower=0.5, upper=1.5)
    image2 = tf.image.random_saturation(image2, lower=0.5, upper=1.5)
    image2 = tf.image.random_hue(image2, max_delta=0.2)
    return tf.stack([image1, image2])
