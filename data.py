import tensorflow as tf

patch_height = 110
patch_width = 110
patch_overlap = 22

def image_to_patches(image):
    patches = tf.extract_image_patches(image, [1, patch_height, patch_width, 1], [1, patch_height - 2 * patch_overlap, patch_width - 2 * patch_overlap, 1], [1, 1, 1, 1], padding='VALID')
    return tf.reshape(patches, [tf.shape(patches)[0] * tf.shape(patches)[1] * tf.shape(patches)[2], patch_height, patch_width, 3])