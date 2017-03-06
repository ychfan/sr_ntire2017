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
    distort = True
    scale_list = [1, 2, 3, 4]
    patch_height = 110
    patch_width = 110
    patch_overlap = 22
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
    hr_patches = image_to_patches(hr_image, patch_height, patch_width, patch_overlap)
    hr_patches_list = [hr_patches] * len(scale_list)
    lr_image_list = []
    for scale in scale_list:
        lr_image = tf.image.resize_bicubic(tf.image.resize_bicubic(hr_image, tf.shape(hr_image)[1:3] / scale), tf.shape(hr_image)[1:3])
        lr_image_list.append(lr_image)
    lr_patches = image_to_patches(tf.concat(lr_image_list, 0), patch_height, patch_width, patch_overlap)
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
    
      
def image_to_patches(image, patch_height, patch_width, patch_overlap):
    patches = tf.extract_image_patches(image, [1, patch_height, patch_width, 1], [1, patch_height - 2 * patch_overlap, patch_width - 2 * patch_overlap, 1], [1, 1, 1, 1], padding='VALID')
    return tf.reshape(patches, [tf.shape(patches)[0] * tf.shape(patches)[1] * tf.shape(patches)[2], patch_height, patch_width, 3])