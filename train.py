import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

import data

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('hr_flist', 'data/hr.flist', 'file_list put the training data.')
flags.DEFINE_string('lr_flist', 'data/lr.flist', 'Directory to put the training data.')
flags.DEFINE_string('model_name', 'model_res', 'Directory to put the training data.')

with tf.Graph().as_default():
    target_batch_staging, source_batch_staging = data.dataset_hr(FLAGS.hr_flist)
    stager = data_flow_ops.StagingArea([tf.float32, tf.float32], shapes=[[None, None, None, 1], [None, None, None, 1]])
    stage = stager.put([target_batch_staging, source_batch_staging])
    target_batch, source_batch = stager.get()
    model_def = __import__(FLAGS.model_name)
    predict_batch = model_def.build_model(source_batch)
    loss = tf.losses.mean_squared_error(target_batch, predict_batch)
    floor = tf.losses.mean_squared_error(target_batch, source_batch)
    learning_rate = tf.Variable(0.01, trainable=False)
    adam_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    sgd_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
    optimizer = adam_optimizer
    
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            sess.run(stage)
            while not coord.should_stop():
                _, _, training_loss, baseline = sess.run([stage, optimizer, loss, floor])
                print baseline, baseline - training_loss
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

    
