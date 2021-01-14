import sys
sys.path.append('..')

import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf

from modules.models import ArcFaceModel
from modules.utils import l2_norm

flags.DEFINE_string('weights', '../checkpoints/e_8_b_40000.ckpt',
                    'path to weights file')
flags.DEFINE_string('output_dir', '../serving', 'directory to saved_model')
flags.DEFINE_integer('version', 1, 'export version')
flags.DEFINE_integer('input_size', 112, 'input images resolution')
flags.DEFINE_string('image', '', 'path to input image')


def main(_argv):
    model = ArcFaceModel(size=FLAGS.input_size, training=False, use_pretrain=False)
    model.load_weights(FLAGS.weights).expect_partial()
    model.summary()

    # Saved path will be 'output_dir/model_name/version'
    saved_path = os.path.join(FLAGS.output_dir, 'arcface', str(FLAGS.version))
    tf.saved_model.save(model, saved_path)
    logging.info("model saved to: {}".format(saved_path))

    model = tf.saved_model.load(saved_path)
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    logging.info(infer.structured_outputs)

    if not FLAGS.image:
        return
    img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    img = tf.image.resize(img, (FLAGS.input_size, FLAGS.input_size))
    img = img / 255.
    img = tf.expand_dims(img, 0)
 
    t1 = time.time()
    outputs = infer(img)
    embeddings = outputs['OutputLayer']
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
