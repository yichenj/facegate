import sys
sys.path.append('..')

import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3.models import (
    YoloV3, YoloV3Tiny
)
from yolov3.dataset import load_yolo_dataset, transform_images
from yolov3.utils import draw_outputs

flags.DEFINE_string('classes', '../data/face.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string(
    'images', '', 'path to images dir')
flags.DEFINE_string('labels', '', 'path to label file')


def main(_argv):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if not FLAGS.images or not FLAGS.labels:
        logging.error('Image directory and label file must be sepcified.')
        return
    dataset = load_yolo_dataset(FLAGS.images, FLAGS.labels, FLAGS.size)
    dataset = dataset.shuffle(512)

    for image, labels in dataset:
        boxes = []
        scores = []
        classes = []
        for x1, y1, x2, y2, label in labels:
            if x1.numpy() == 0 and x2.numpy() == 0:
                continue

            boxes.append((x1, y1, x2, y2))
            scores.append(1)
            classes.append(label)
        nums = [len(boxes)]
        boxes = [boxes]
        scores = [scores]
        classes = [classes]

        logging.info('labels:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))
        img = cv2.cvtColor(np.uint8(image.numpy()), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imshow('Dataset Visualizer', img)
        key = cv2.waitKey(0)
        if key == 27:
          return


if __name__ == '__main__':
    app.run(main)
