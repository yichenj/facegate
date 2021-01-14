from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf

from arcface.models import ArcFaceModel
from arcface.utils import l2_normalize


flags.DEFINE_string('image', '', 'path to input image')
flags.DEFINE_string('weights', './checkpoints/e_8_b_40000.ckpt', 'path to weights file')
flags.DEFINE_integer('input_size', 112, 'input image resolution')
flags.DEFINE_float('threshold', 0.5, 'threshold to decide same person')


TEST_IMAGES = {
    'Angelababy': './photo/ab.png',
    '杨幂': './photo/yangmi.png',
    '迪丽热巴': './photo/dilireba.png',
}


def distance(a, b):
    # Euclidean distance
    # return np.linalg.norm(a - b)
    a = a.numpy().reshape(-1)
    b = b.numpy().reshape(-1)
    # Cosine distance, ||a|| and ||b|| is one because embeddings are normalized.
    # No need to compute np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.dot(a, b)


def get_embeddings(model, image_path):
    img = tf.image.decode_image(open(image_path, 'rb').read(), channels=3)
    img = tf.image.resize(img, (FLAGS.input_size, FLAGS.input_size))
    img = img / 255.
    img = tf.expand_dims(img, 0)
    embeds = l2_normalize(model(img))
    return embeds


def init_test(model):
    corpus = {}
    for k, v in TEST_IMAGES.items():
        corpus[k] = get_embeddings(model, v)
    return corpus


def main(_argv):
    model = ArcFaceModel(size=FLAGS.input_size,
                         training=False, use_pretrain=False)
    model.load_weights(FLAGS.weights).expect_partial()
    model.summary()

    corpus = init_test(model)

    if not FLAGS.image:
        print("Image path must be specified.")
        return 1
    embeds = get_embeddings(model, FLAGS.image)

    print('\n\nRecognizing......\n\n')
    for name, database_embeds in corpus.items():
        d = distance(database_embeds, embeds)
        print(name, d)
        if d > FLAGS.threshold:
            print('Welcome, ' + name + ' [distance = %f].\n\n' % d)
            return
    print('Not recognized..\n\n')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
