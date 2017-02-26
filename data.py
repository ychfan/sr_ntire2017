import tensorflow as tf

def dataset(hr_flist, lr_flist):
    scale = 2
    with open(hr_flist) as f:
        hr_filename_list = f.read().splitlines()
    with open(lr_flist) as f:
        lr_filename_list = f.read().splitlines()
    filename_queue = tf.train.string_input_producer([hr_filename_list, lr_filename_list])
    reader = tf.WholeFileReader()
    _, hr_image_file = reader.read(filename_queue[0])
    _, lr_image_file = reader.read(filename_queue[1])
    hr_image = tf.image.decode_image(hr_image_file, channels=3)
    lr_image = tf.image.decode_image(lr_image_file, channels=3)
    hr_grayscale = tf.image.rgb_to_grayscale(hr_image)
    lr_grayscale = tf.image.rgb_to_grayscale(lr_image)
    return hr_grayscale, lr_grayscale
    

def dataset_hr(hr_flist):
    scale = 2
    seed = 0
    cropped_height = 300
    cropped_width = 400
    boundary = 6
    with open(hr_flist) as f:
        hr_filename_list = f.read().splitlines()
    filename_queue = tf.train.string_input_producer(hr_filename_list)
    reader = tf.WholeFileReader()
    _, image_file = reader.read(filename_queue)
    hr_image = tf.image.decode_image(image_file, channels=3)
    hr_image_batch = tf.expand_dims(hr_image, 0)
    lr_image_batch = tf.image.resize_bicubic(hr_image_batch, tf.shape(hr_image)[0:2] / scale)
    hr_grayscale_batch = tf.image.rgb_to_grayscale(hr_image_batch)
    lr_grayscale_batch = tf.image.resize_bicubic(tf.image.rgb_to_grayscale(lr_image_batch), tf.shape(hr_image)[0:2])
    hr_grayscale_batch = tf.cast(hr_grayscale_batch, tf.float32) * (1. / 255)
    lr_grayscale_batch = tf.cast(lr_grayscale_batch, tf.float32) * (1. / 255)
    hr_grayscale_batch = tf.random_crop(hr_grayscale_batch, [1, cropped_height + 2 * boundary, cropped_width + 2 * boundary, 1], seed=seed)
    lr_grayscale_batch = tf.random_crop(lr_grayscale_batch, [1, cropped_height + 2 * boundary, cropped_width + 2 * boundary, 1], seed=seed)
    hr_grayscale_batch = tf.slice(hr_grayscale_batch, [0, boundary, boundary, 0], [-1, cropped_height, cropped_width, -1])
    lr_grayscale_batch = tf.slice(lr_grayscale_batch, [0, boundary, boundary, 0], [-1, cropped_height, cropped_width, -1])
    return hr_grayscale_batch, lr_grayscale_batch
      
      
