import tensorflow as tf
import util

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_name', 'data_residual', 'Directory to put the training data.')
flags.DEFINE_string('hr_flist', 'flist/hr_val.flist', 'file_list put the training data.')
flags.DEFINE_string('lr_flist', 'flist/lrX2_val.flist', 'Directory to put the training data.')
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
    filename_queue = tf.train.string_input_producer(lr_filename_list, num_epochs=2, shuffle=False)
    reader = tf.WholeFileReader()
    _, image_file = reader.read(filename_queue)
    lr_image = tf.image.decode_image(image_file, channels=3)
    lr_image = tf.image.convert_image_dtype(lr_image, tf.float32)
    lr_image = tf.expand_dims(lr_image, 0)
    lr_image_shape = tf.shape(lr_image)[1:3]
    hr_image_shape = lr_image_shape * FLAGS.scale
    if (data.resize):
        lr_image = util.resize_func(lr_image, hr_image_shape)
        lr_image = tf.reshape(lr_image, [1, hr_image_shape[0], hr_image_shape[1], 3])
    else:
        lr_image = tf.reshape(lr_image, [1, lr_image_shape[0], lr_image_shape[1], 3])
    lr_image_padded = util.pad_boundary(lr_image)
    hr_image = model.build_model(lr_image_padded, FLAGS.scale, training=False, reuse=False)
    hr_image = util.crop_center(hr_image, hr_image_shape)
    if (data.residual):
        if (data.resize):
            hr_image += lr_image
        else:
            hr_image += util.resize_func(lr_image, hr_image_shape)
    hr_image = hr_image * tf.uint8.max + 0.5
    hr_image = tf.saturate_cast(hr_image, tf.uint8)
    hr_image = tf.reshape(hr_image, [hr_image_shape[0], hr_image_shape[1], 3])
    hr_image = tf.image.encode_png(hr_image)
    
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    saver = tf.train.Saver()
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
                img = sess.run([hr_image])
                with open(hr_filename, 'w') as f:
                    f.write(img[0])
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()