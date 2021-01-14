import time
import os
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3.models import (
    YoloV3, YoloV3Tiny
)
from yolov3.dataset import transform_images
from yolov3.utils import draw_outputs

flags.DEFINE_string('classes', './data/face.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
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

    if not FLAGS.image:
        login.error('Detect image must be specified')
        return 1
    elif os.path.isdir(FLAGS.image):
        detect_images = [os.path.join(FLAGS.image, x) for x in os.listdir(FLAGS.image) if x[-3:] == 'jpg']
    else:
        detect_images = [FLAGS.image] 
 
    for image in detect_images:
        img_raw = tf.image.decode_image(
            open(image, 'rb').read(), channels=3)

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('detections:')
        if nums[0].numpy() == 0:
            continue

        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imshow(image, img)
        cv2.waitKey(0)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
