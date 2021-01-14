import tensorflow as tf


def parse_ciasia_facev5_crop_dataset(line, image_path):
    line = tf.strings.split(line, ',')
    image_file = tf.strings.reduce_join([image_path, line[0]], separator='/')
    label = tf.strings.to_number(line[1], out_type=tf.int32)
    return image_file, label


def parse_ciasia_facev5_crop_image(image, label, size):
    image = tf.io.read_file(image)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (size, size))
    image = image / 255
    return (image, label), label


def image_preprocessing(x, y):
    x = x[0]
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_saturation(x, 0.6, 1.4)
    x = tf.image.random_brightness(x, 0.4)
    return (x, y), y


def load_ciasia_facev5_crop_dataset(image_path, label_file, input_size, shuffle=False, shuffle_buffer_size=512):
    dataset = tf.data.TextLineDataset(label_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(lambda x: parse_ciasia_facev5_crop_dataset(x, image_path))
    dataset = dataset.map(lambda x,y: parse_ciasia_facev5_crop_image(x, y, input_size))
    return dataset
        

def get_dataset_size(dataset_labels):
    all_classes = set()
    with open(dataset_labels, 'r') as f:
        for i, l in enumerate(f):
            image, class_name = l.split(',')
            all_classes.add(class_name)
        return i + 1, len(all_classes)
