import tensorflow as tf

resize_func = tf.image.resize_nearest_neighbor

def image_to_patches(image, scale=1):
    patch_height = 108 / scale
    patch_width = 108 / scale
    patch_overlap = 12 / scale
    patches = tf.extract_image_patches(image, [1, patch_height, patch_width, 1], [1, patch_height - 2 * patch_overlap, patch_width - 2 * patch_overlap, 1], [1, 1, 1, 1], padding='VALID')
    return tf.reshape(patches, [tf.shape(patches)[0] * tf.shape(patches)[1] * tf.shape(patches)[2], patch_height, patch_width, 3])

def crop_center(image, target_shape):
    origin_shape = tf.shape(image)[1:3]
    return tf.slice(image, [0, (origin_shape[0] - target_shape[0]) / 2, (origin_shape[1] - target_shape[1]) / 2, 0], [-1, target_shape[0], target_shape[1], -1])

def crop_by_pixel(x, num):
    shape = tf.shape(x)[1:3]
    return tf.slice(x, [0, num, num, 0], [-1, shape[0] - 2 * num, shape[1] - 2 * num, -1])

def pad_boundary(image, boundary_size=15):
    return tf.pad(image, [[0, 0], [boundary_size, boundary_size], [boundary_size, boundary_size], [0, 0]], mode="SYMMETRIC")