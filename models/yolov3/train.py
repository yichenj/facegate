from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
)
from yolov3.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3.utils import freeze_all
import yolov3.dataset as dataset

flags.DEFINE_string('train_images', '', 'path to training set')
flags.DEFINE_string('train_labels', '', 'path to training set label file')
flags.DEFINE_string('val_images', '', 'path to validation set')
flags.DEFINE_string('val_labels', '', 'path to validation set label file')
flags.DEFINE_string('classes', './data/face.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager'],
                  'fit: model.fit, '
                  'eager: custom GradientTape')
flags.DEFINE_integer('epochs', 50, 'number of epochs')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_enum('freeze', 'none', ['none', 'darknet', 'no_output'],
                  'Freeze none, '
                  'Freeze darknet layers, '
                  'Freeze all but output layer')
flags.DEFINE_string('weights', None,
                    'path to weights file')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file'
                     'when do transfer learning with different number of classes task, '
                     'e.g. darknet weights_num_classes = 80')
flags.DEFINE_enum('transfer', 'no_output', ['no_output', 'darknet'],
                  'Transfer darknet, '
                  'Transfer all but output')


def main(_argv):
    num_classes = len([c.strip() for c in open(FLAGS.classes).readlines()])
    logging.info('classes loaded, number of classes = %d' % num_classes)

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # TensorFlow 2.0/2.1 has a bug when putting finite dateset to model.fit().
    # Walk around by using dataset.repeat() and give the number of steps to model fit.
    # Refer to https://github.com/tensorflow/tensorflow/issues/31509
    training_set_size = dataset.get_dataset_size(FLAGS.train_labels)
    steps_for_train = training_set_size // FLAGS.batch_size
    steps_for_val = dataset.get_dataset_size(FLAGS.val_labels) // FLAGS.batch_size
    logging.info('Training in %d steps, validation in %d steps' % (steps_for_train, steps_for_val))

    if not (FLAGS.train_images and FLAGS.train_labels):
      logging.error('Training images and labels must be specified.')
      return
    train_dataset = dataset.load_yolo_dataset(
        FLAGS.train_images, FLAGS.train_labels, FLAGS.size, shuffle=True, shuffle_buffer_size=training_set_size)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(FLAGS.batch_size, drop_remainder=True)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    if not (FLAGS.val_images and FLAGS.val_labels):
      logging.error('Validation images and labels must be specified.')
      return
    val_dataset = dataset.load_yolo_dataset(
        FLAGS.val_images, FLAGS.val_labels, FLAGS.size)
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.batch(FLAGS.batch_size, drop_remainder=True)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    val_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    # Configure the model
    # Weights configuration
    if FLAGS.weights_num_classes is None:
        if FLAGS.weights:
            model.load_weights(FLAGS.weights)
    else:
        # Transfer learning with incompatible number of classes
        assert FLAGS.weights

        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or num_classes)
        else:
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or num_classes)
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())

    # Freeze layers
    if FLAGS.freeze == 'none':
        pass
    elif FLAGS.freeze == 'darknet':
        freeze_all(model.get_layer('yolo_darknet'))
    elif FLAGS.freeze == 'no_output':
        # freeze all but output layers
        for l in model.layers:
            if not l.name.startswith('yolo_output'):
                freeze_all(l)
    model.summary()

    # Configure training process
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=num_classes, ignore_thresh=0.5)
            for mask in anchor_masks]

    if FLAGS.mode == 'eager':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            batch = 0
            for images, labels in train_dataset:
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

                batch += 1
                if batch == steps_for_train:
                    break

            batch = 0
            for images, labels in val_dataset:
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

                batch += 1
                if batch == steps_for_val:
                    break

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(
                'checkpoints/yolov3_train_{}.tf'.format(epoch))
    else:
        model.compile(optimizer=optimizer, loss=loss)

        callbacks = [
            ReduceLROnPlateau(verbose=1),
#            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True, save_best_only=True),
        ]

        model.fit(train_dataset,
                  epochs=FLAGS.epochs,
                  steps_per_epoch=steps_for_train,
                  callbacks=callbacks,
                  validation_data=val_dataset,
                  validation_steps=steps_for_val)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
