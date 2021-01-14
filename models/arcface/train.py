from absl import app, flags, logging
from absl.flags import FLAGS

import os
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint

from arcface.models import ArcFaceModel
from arcface.losses import SoftmaxLoss
import arcface.dataset as dataset

flags.DEFINE_string('train_images', '', 'path to training set')
flags.DEFINE_string('train_labels', '', 'path to training set label file')
flags.DEFINE_string('val_images', '', 'path to validation set')
flags.DEFINE_string('val_labels', '', 'path to validation set label file')
flags.DEFINE_integer('input_size', 112, 'image size')
flags.DEFINE_integer('epochs', 50, 'number of epochs')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
flags.DEFINE_float('lr_decay_epoches', 2.0, 'learning rate decay after the number of epoches')
flags.DEFINE_float('lr_decay_rate', 0.9, 'learning rate decay rate')
flags.DEFINE_string('weights', None,
                     'path to weights file')


def main(_):
    if not (FLAGS.train_images and FLAGS.train_labels):
        logging.error('Training images and labels must be specified.')
        return
    if not (FLAGS.val_images and FLAGS.val_labels):
        logging.error('Validation images and labels must be specified.')
        return

    # TensorFlow 2.0/2.1 has a bug when putting finite dateset to model.fit().
    # Walk around by using dataset.repeat() and give the number of steps to model fit.
    # Refer to https://github.com/tensorflow/tensorflow/issues/31509
    training_set_size, num_of_classes = dataset.get_dataset_size(FLAGS.train_labels)
    logging.info('Load %d images of %d classes for training set' % (training_set_size, num_of_classes))
    steps_for_train = training_set_size // FLAGS.batch_size
    val_set_size, _ = dataset.get_dataset_size(FLAGS.val_labels) 
    steps_for_val = val_set_size // FLAGS.batch_size
    logging.info('Load %d images of %d classes for validation set' % (val_set_size, num_of_classes))
    logging.info('Training in %d steps, validation in %d steps' % (steps_for_train, steps_for_val))

    # Create model
    model = ArcFaceModel(size=FLAGS.input_size,
                         num_classes=num_of_classes,
                         training=True)
    model.summary()

    # Load dataset
    train_dataset = dataset.load_ciasia_facev5_crop_dataset(FLAGS.train_images, FLAGS.train_labels, FLAGS.input_size, shuffle=True, shuffle_buffer_size=training_set_size)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(FLAGS.batch_size, drop_remainder=True)
    train_dataset = train_dataset.map(map_func=dataset.image_preprocessing)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = dataset.load_ciasia_facev5_crop_dataset(FLAGS.val_images, FLAGS.val_labels, FLAGS.input_size)
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.batch(FLAGS.batch_size, drop_remainder=True)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Load weights
    if FLAGS.weights:
        logging.info('Loading weights %s' % FLAGS.weights)
        model_pretrained = ArcFaceModel(size=FLAGS.input_size, num_classes=num_of_classes, training=False)
        model_pretrained.load_weights(FLAGS.weights).expect_partial()
        for l in model_pretrained.layers:
            model.get_layer(l.name).set_weights(l.get_weights())

    # Initialize hyper parameters.
    initial_learning_rate = FLAGS.learning_rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=FLAGS.lr_decay_epoches * steps_for_train,
        decay_rate=FLAGS.lr_decay_rate,
        staircase=True)
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=lr_schedule, rho=0.9, momentum=0.9)
    loss_fn = SoftmaxLoss()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Begin training.
    cp_callback = ModelCheckpoint(
        'checkpoints/arcface_train_{epoch}.ckpt',
        verbose=1, save_best_only=True,
        save_weights_only=True)
    model.fit(train_dataset,
              epochs=FLAGS.epochs,
              steps_per_epoch=steps_for_train,
              callbacks=[cp_callback],
              # WR for TF2.0 bug. Must specify shuffle=False when using a generator,
              # otherwise fit() drains the entire generator when shuffle=True(by default).
              # Refer to https://github.com/tensorflow/tensorflow/issues/33024.
              shuffle=False,
              validation_data=val_dataset,
              validation_steps=steps_for_val)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
