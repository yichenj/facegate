import os
import random

from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_string('data', './CASIA-FaceV5-Crop', 'path to the CASIA-FaceV5-Crop dataset which needs to be convert.')


def get_images_per_category(dataset_dir, category):
    all_images = []
    for each in os.listdir(os.path.join(dataset_dir, category)):
        if not each[-3:] == 'bmp':
            continue
        all_images.append(os.path.join(category, each))
    return all_images


def main(_argv):
    training_set_file, val_set_file = 'casia_facev5_train.txt', 'casia_facev5_val.txt'
    training_set = open(training_set_file, 'w+')
    val_set = open(val_set_file, 'w+')

    dataset_dir = FLAGS.data
    for category in os.listdir(dataset_dir):
        if not os.path.isdir(os.path.join(dataset_dir, category)):
            continue

        images = get_images_per_category(dataset_dir, category)
        if len(images) <= 0:
            continue

        random.shuffle(images)
        val_set.write('%s,%d\n' % (images[0], int(category)))
        for image in images[1:]:
            training_set.write('%s,%d\n' % (image, int(category)))

    training_set.close()
    val_set.close()
    return 0


if __name__ == '__main__':
    app.run(main)
