import tensorflow as tf
import math
import util

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_name', 'data_residual', 'Directory to put the training data.')
flags.DEFINE_string('hr_flist', 'flist/hr_val.flist', 'file_list put the training data.')
flags.DEFINE_string('lr_flist', 'flist/lrX2_bicubic_val.flist', 'Directory to put the training data.')
flags.DEFINE_integer('scale', '2', 'batch size for training')
flags.DEFINE_string('model_name', 'model_conv', 'Directory to put the training data.')
flags.DEFINE_string('model_file', 'tmp/model_conv', 'Directory to put the training data.')

data = __import__(FLAGS.data_name)
model = __import__(FLAGS.model_name)
if (data.resize == model.upsample):
    print "Config Error"
    quit()

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
    lr_image_shape = tf.shape(lr_image)[1:3]
    hr_image_shape = tf.shape(hr_image)[1:3]
    if (data.resize):
        lr_image = util.resize_func(lr_image, hr_image_shape)
        lr_image = tf.reshape(lr_image, [1, hr_image_shape[0], hr_image_shape[1], 3])
    else:
        lr_image = tf.reshape(lr_image, [1, lr_image_shape[0], lr_image_shape[1], 3])
    lr_image_padded = util.pad_boundary(lr_image)
    sr_image = model.build_model(lr_image_padded - 0.5, FLAGS.scale, training=False, reuse=False)
    sr_image = util.crop_center(sr_image, hr_image_shape)
    if (data.residual):
        if (data.resize):
            sr_image += lr_image
        else:
            sr_image += util.resize_func(lr_image, hr_image_shape)
    sr_image = sr_image * tf.uint8.max + 0.5
    sr_image = tf.saturate_cast(sr_image, tf.uint8)
    sr_image = tf.cast(sr_image, tf.float32)
    sr_image = sr_image * (1.0 / tf.uint8.max)
    sr_image = util.crop_by_pixel(sr_image, FLAGS.scale + 6)
    hr_image = util.crop_by_pixel(hr_image, FLAGS.scale + 6)
    error = tf.losses.mean_squared_error(hr_image, sr_image)
    
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    saver = tf.train.Saver()
    error_acc = .0
    psnr_acc = .0
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
                psnr_per_image = -10.0 * math.log10(error_per_image)
                print error_per_image, psnr_per_image
                error_acc += error_per_image
                psnr_acc += psnr_per_image
                acc += 1
        except tf.errors.OutOfRangeError:
            print('Done validation -- epoch limit reached')
        finally:
            coord.request_stop()
        print error_acc / acc, psnr_acc / acc