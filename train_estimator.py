import tensorflow as tf
from tensorflow.python.lib.io import file_io
import util

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_name', 'data_residual',
                    'Directory to put the training data.')
flags.DEFINE_string('hr_flist', 'flist/hr.flist',
                    'file_list put the training data.')
flags.DEFINE_string('lr_flist', 'flist/lrX2_bicubic.flist',
                    'Directory to put the training data.')
flags.DEFINE_string('val_hr_flist', 'flist/hr_val.flist',
                    'file_list put the training data.')
flags.DEFINE_string('val_lr_flist', 'flist/lrX2_bicubic_val.flist',
                    'Directory to put the training data.')
flags.DEFINE_integer('scale', '2', 'batch size for training')
flags.DEFINE_string('model_name', 'model_resnet_up',
                    'Directory to put the training data.')
flags.DEFINE_string('model_dir', 'tmp/model_conv',
                    'Directory to put the training data.')
flags.DEFINE_float('learning_rate', '0.001', 'Learning rate for training')
flags.DEFINE_integer('batch_size', '32', 'batch size for training')
flags.DEFINE_boolean('mem_growth', True, 'If true, use gpu memory on demand.')

from importlib import import_module
data = import_module(FLAGS.data_name)
model = import_module(FLAGS.model_name)
if data.resize == model.upsample:
    print("Config Error")
    quit()


def model_fn(features, labels, mode, params):
    del params
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = model.build_model(
            features - 0.5, FLAGS.scale, training=False, reuse=False)
        if data.residual:
            predictions += labels
        predictions = predictions * tf.uint8.max + 0.5
        predictions = tf.saturate_cast(predictions, tf.uint8)
        predictions = tf.reshape(
            predictions, [hr_image_shape[0], hr_image_shape[1], 3])
        loss = None
        train_op = None
        eval_metric_ops = None
    else:
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        predict_batch = model.build_model(
            features, FLAGS.scale, training=is_training, reuse=None)
        target_cropped_batch = util.crop_center(labels,
                                                tf.shape(predict_batch)[1:3])
        loss = tf.losses.mean_squared_error(
            target_cropped_batch, predict_batch)
        tf.summary.scalar('loss', loss)
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            loss,
            global_step=tf.train.get_global_step(),
        )
        eval_metric_ops = {
            "mse": tf.metrics.mean_squared_error(target_cropped_batch, predict_batch)
        }
        predictions = None
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
    )


def input_fn(lr_flist, hr_flist, mode):
    with tf.device('/cpu:0'):
        if mode == tf.estimator.ModeKeys.PREDICT:
            with file_io.FileIO(lr_flist, "r") as f:
                lr_filename_list = f.read().splitlines()
            filename_queue = tf.train.slice_input_producer(
                [lr_filename_list], num_epochs=1, shuffle=False)
            image_file = tf.read_file(filename_queue[0])
            lr_image = tf.image.decode_image(image_file, channels=3)
            lr_image = tf.image.convert_image_dtype(lr_image, tf.float32)
            lr_image = tf.expand_dims(lr_image, 0)
            lr_image_shape = tf.shape(lr_image)[1:3]
            hr_image_shape = lr_image_shape * FLAGS.scale
            if data.resize:
                lr_image = util.resize_func(lr_image, hr_image_shape)
                lr_image = tf.reshape(
                    lr_image, [1, hr_image_shape[0], hr_image_shape[1], 3])
            else:
                lr_image = tf.reshape(
                    lr_image, [1, lr_image_shape[0], lr_image_shape[1], 3])
            lr_image_padded = util.pad_boundary(lr_image)
            if data.residual:
                if data.resize:
                    hr_image = lr_image
                else:
                    hr_image = util.resize_func(lr_image, hr_image_shape)
            else:
                hr_image = None
            return lr_image, hr_image

        else:
            target_patches, source_patches = data.dataset(
                hr_flist, lr_flist, FLAGS.scale)
            if mode == tf.estimator.ModeKeys.TRAIN:
                target_batch, source_batch = tf.train.shuffle_batch(
                    [target_patches, source_patches],
                    FLAGS.batch_size,
                    32768,
                    8192,
                    num_threads=4,
                    enqueue_many=True)
            else:
                target_batch, source_batch = tf.train.batch(
                    [target_patches, source_patches],
                    FLAGS.batch_size,
                    num_threads=4,
                    enqueue_many=True,
                    allow_smaller_final_batch=True)
            return source_batch, target_batch


def experiment_fn(run_config, hparams):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams,
    )
    return tf.contrib.learn.Experiment(
        estimator,
        train_input_fn=lambda: input_fn(
            FLAGS.val_lr_flist, FLAGS.val_hr_flist, tf.estimator.ModeKeys.TRAIN),
        eval_input_fn=lambda: input_fn(
            FLAGS.val_lr_flist, FLAGS.val_hr_flist, tf.estimator.ModeKeys.EVAL),
        train_steps=None,
        eval_steps=None,
        train_steps_per_iteration=100000,
    )


def main(unused_argv):
    del unused_argv  # Unused

    tf.logging.set_verbosity(tf.logging.INFO)

    gpu_options = tf.GPUOptions(
        allow_growth=True,
        force_gpu_compatible=True,
    )
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=gpu_options,
    )
    sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    tf.contrib.learn.learn_runner.run(
        experiment_fn,
        run_config=tf.contrib.learn.RunConfig(
            model_dir=FLAGS.model_dir,
            session_config=sess_config,
        ),
        schedule="continuous_train_and_eval",
    )


if __name__ == "__main__":
    tf.app.run()
