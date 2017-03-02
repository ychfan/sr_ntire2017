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
    scale_list = [2, 3, 4]
    patch_height = 110
    patch_width = 110
    patch_overlap = 10
    with open(hr_flist) as f:
        hr_filename_list = f.read().splitlines()
    filename_queue = tf.train.string_input_producer(hr_filename_list, num_epochs=1)
    reader = tf.WholeFileReader()
    _, image_file = reader.read(filename_queue)
    hr_image = tf.image.decode_image(image_file, channels=3)
    hr_image = tf.expand_dims(hr_image, 0)
    hr_image = tf.image.convert_image_dtype(hr_image, tf.float32)
    hr_grayscale = tf.image.rgb_to_grayscale(hr_image)
    hr_grayscale_patches = image_to_patches(hr_grayscale, patch_height, patch_width, patch_overlap)
    hr_grayscale_patches_list = []
    lr_grayscale_patches_list = []
    for scale in scale_list:
        lr_image = tf.image.resize_bicubic(hr_image, tf.shape(hr_image)[1:3] / scale)
        lr_grayscale = tf.image.resize_bicubic(tf.image.rgb_to_grayscale(lr_image), tf.shape(hr_image)[1:3])
        lr_grayscale_patches = image_to_patches(lr_grayscale, patch_height, patch_width, patch_overlap)
        hr_grayscale_patches_list.append(hr_grayscale_patches)
        lr_grayscale_patches_list.append(lr_grayscale_patches)
    hr_grayscale_patches = tf.concat(hr_grayscale_patches_list, 0)
    lr_grayscale_patches = tf.concat(lr_grayscale_patches_list, 0)
    return hr_grayscale_patches, lr_grayscale_patches
      
def image_to_patches(image, patch_height, patch_width, patch_overlap):
    patches = tf.extract_image_patches(image, [1, patch_height, patch_width, 1], [1, patch_height - 2 * patch_overlap, patch_width - 2 * patch_overlap, 1], [1, 1, 1, 1], padding='VALID')
    return tf.reshape(patches, [tf.shape(patches)[1] * tf.shape(patches)[2], patch_height, patch_width, 1])