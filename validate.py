import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('hr_flist', 'flist/hr.flist', 'file_list put the training data.')
flags.DEFINE_string('lr_flist', 'flist/lrX2.flist', 'Directory to put the training data.')
flags.DEFINE_string('model_name', 'model_res_pre_act', 'Directory to put the training data.')
flags.DEFINE_string('model_file', 'tmp/model_res_pre_act-data_naive_nn-try', 'Directory to put the training data.')

def crop_center(image, target_shape):
    origin_shape = tf.shape(image)[1:3]
    return tf.slice(image, [0, (origin_shape[0] - target_shape[0]) / 2, (origin_shape[1] - target_shape[1]) / 2, 0], [-1, target_shape[0], target_shape[1], -1])

with tf.Graph().as_default():
    with open(FLAGS.hr_flist) as f:
        hr_filename_list = f.read().splitlines()
    with open(FLAGS.lr_flist) as f:
        lr_filename_list = f.read().splitlines()
    filename_queue = tf.train.slice_input_producer([hr_filename_list, lr_filename_list], num_epochs=2, shuffle=False)
    hr_image_file = tf.read_file(filename_queue[0])
    lr_image_file = tf.read_file(filename_queue[1])
    hr_image = tf.image.decode_image(hr_image_file, channels=3)
    lr_image = tf.image.decode_image(lr_image_file, channels=3)
    hr_image = tf.image.convert_image_dtype(hr_image, tf.float32)
    lr_image = tf.image.convert_image_dtype(lr_image, tf.float32)
    hr_image = tf.expand_dims(hr_image, 0)
    lr_image = tf.expand_dims(lr_image, 0)
    hr_image_shape = tf.shape(hr_image)[1:3]
    lr_image = tf.image.resize_nearest_neighbor(lr_image, hr_image_shape)
    lr_image = tf.reshape(lr_image, [1, hr_image_shape[0], hr_image_shape[1], 3])
    boundary_size = 15
    lr_image_padded = tf.pad(lr_image, [[0, 0], [boundary_size, boundary_size], [boundary_size, boundary_size], [0, 0]], mode="SYMMETRIC")
    model_def = __import__(FLAGS.model_name)
    res = model_def.build_model(lr_image_padded, False)
    lr_image = lr_image + crop_center(res, hr_image_shape)
    error = tf.losses.mean_squared_error(hr_image, lr_image)
    
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    saver = tf.train.Saver()
    error_acc = .0
    acc = 0
    with tf.Session() as sess:
        sess.run(init_local)
        if (tf.gfile.Exists(FLAGS.model_file) or tf.gfile.Exists(FLAGS.model_file + '.index')):
            saver.restore(sess, FLAGS.model_file)
            print 'Model restored from ' + FLAGS.model_file
        else:
            print 'Model not found'
            exit()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for hr_filename in hr_filename_list:
                error_per_image = sess.run(error)
                print error_per_image
                error_acc += error_per_image
                acc += 1
        except tf.errors.OutOfRangeError:
            print('Done validation -- epoch limit reached')
        finally:
            coord.request_stop()
        print error_acc / acc