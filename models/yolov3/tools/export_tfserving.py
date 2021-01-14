import sys
sys.path.append('..')

import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3.models import (
    YoloV3, YoloV3Tiny
)
from yolov3.dataset import transform_images

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest

flags.DEFINE_string('weights', '../checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('output_dir', '../serving', 'directory to saved_model')
flags.DEFINE_integer('version', 1, 'export version')
flags.DEFINE_string('classes', '../data/face.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', '', 'path to input image')


def main(_argv):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=len(class_names))
    else:
        yolo = YoloV3(classes=len(class_names))

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    # Saved path will be 'output_dir/model_name/version'
    saved_path = os.path.join(FLAGS.output_dir, 'yolov3', str(FLAGS.version))
    tf.saved_model.save(yolo, saved_path)
    logging.info("model saved to: {}".format(saved_path))

    model = tf.saved_model.load(saved_path)
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    logging.info(infer.structured_outputs)

    if not FLAGS.image:
        return

    img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    outputs = infer(img)
    boxes, scores, classes, nums = outputs["yolo_nms"], outputs[
        "yolo_nms_1"], outputs["yolo_nms_2"], outputs["yolo_nms_3"]
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           scores[0][i].numpy(),
                                           boxes[0][i].numpy()))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
