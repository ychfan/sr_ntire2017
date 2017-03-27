import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_name', 'data_naive_bc', 'Directory to put the training data.')
flags.DEFINE_string('hr_flist', 'flist/hr.flist', 'file_list put the training data.')
flags.DEFINE_string('lr_flist', 'flist/lrX2.flist', 'Directory to put the training data.')
flags.DEFINE_string('model_name', 'model_res', 'Directory to put the training data.')
flags.DEFINE_string('model_file_in', 'tmp/model_res', 'Directory to put the training data.')
flags.DEFINE_string('model_file_out', 'tmp/model_res', 'Directory to put the training data.')
flags.DEFINE_float('learning_rate', '0.001', 'Learning rate for training')
flags.DEFINE_boolean('adam', True, 'Use adam optimizer for training')
flags.DEFINE_integer('batch_size', '32', 'batch size for training')

def crop_center(image, target_shape):
    origin_shape = tf.shape(image)[1:3]
    return tf.slice(image, [0, (origin_shape[0] - target_shape[0]) / 2, (origin_shape[1] - target_shape[1]) / 2, 0], [-1, target_shape[0], target_shape[1], -1])

with tf.Graph().as_default():
    data = __import__(FLAGS.data_name)
    target_patches, source_patches = data.dataset(FLAGS.hr_flist, FLAGS.lr_flist)
    target_batch_staging, source_batch_staging = tf.train.shuffle_batch([target_patches, source_patches], FLAGS.batch_size, 32768, 8192, num_threads=4, enqueue_many=True)
    stager = data_flow_ops.StagingArea([tf.float32, tf.float32], shapes=[[None, None, None, 3], [None, None, None, 3]])
    stage = stager.put([target_batch_staging, source_batch_staging])
    target_batch, source_batch = stager.get()
    model_def = __import__(FLAGS.model_name)
    res_batch = model_def.build_model(source_batch, True)
    target_cropped_batch = crop_center(target_batch, tf.shape(res_batch)[1:3])
    source_cropped_batch = crop_center(source_batch, tf.shape(res_batch)[1:3])
    loss = tf.losses.mean_squared_error(target_cropped_batch, res_batch + source_cropped_batch)
    floor = tf.losses.mean_squared_error(target_cropped_batch, source_cropped_batch)
    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    adam_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    sgd_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
    if FLAGS.adam:
        optimizer = adam_optimizer
    else:
        optimizer = sgd_optimizer
    
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    saver = tf.train.Saver()
    loss_acc = .0
    baseline_acc = .0
    acc = 0
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init_local)
        if (tf.gfile.Exists(FLAGS.model_file_out) or tf.gfile.Exists(FLAGS.model_file_out + '.index')):
            print 'Model exists'
            quit()
        if (tf.gfile.Exists(FLAGS.model_file_in) or tf.gfile.Exists(FLAGS.model_file_in + '.index')):
            saver.restore(sess, FLAGS.model_file_in)
            print 'Model restored from ' + FLAGS.model_file_in
        else:
            sess.run(init)
            print 'Model initialized'
        learning_rate.assign(FLAGS.learning_rate)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            sess.run(stage)
            while not coord.should_stop():
                _, _, training_loss, baseline = sess.run([stage, optimizer, loss, floor])
                print baseline, baseline - training_loss
                loss_acc += training_loss
                baseline_acc += baseline
                acc += 1
                if (acc % 100000 == 0):
                    saver.save(sess, FLAGS.model_file_out + '-' + str(acc))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        print loss_acc / acc, baseline_acc / acc
        saver.save(sess, FLAGS.model_file_out)
        print 'Model saved to ' + FLAGS.model_file_out