import time
import os

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2

flags.DEFINE_string('data_path', '.', 'path to the wider face dataset  which needs to be convert.')
flags.DEFINE_integer('blur_level', 0, 'ignore blur level higher than this threshold')
flags.DEFINE_integer('max_boxes', 100, 'ignore bounding boxes higher than this threshold')


def convert_to_yolo_input(wider_face_label, image_shape):
    x1, y1, w, h = wider_face_label
    image_width, image_height = int(image_shape[0]), int(image_shape[1])
    x_min = int(x1) / image_width
    y_min = int(y1) / image_height
    x_max = (int(x1) + int(w)) / image_width
    y_max = (int(y1) + int(h)) / image_height
    # We have only one class: face
    class_id = 0
    return str(x_min), str(y_min), str(x_max), str(y_max), str(class_id)


def get_image_shape(image_path):
    image = cv2.imread(image_path)
    rows, columns = image.shape[:2]
    width, height = columns, rows
    return width, height


def build_yolo_label(image_info, image_shape):
    image_path, bbx_info = image_info
    all_bbx_list = []
    for each in bbx_info:
        x_min, y_min, x_max, y_max, class_id = convert_to_yolo_input(each, image_shape)
        all_bbx_list.append(' '.join((x_min, y_min, x_max, y_max, class_id)))
    return image_path + ',' + ','.join(all_bbx_list) + os.linesep


def parse_bbx_info(file):
    raw = file.readline().rstrip(os.linesep)
    x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose, _ = raw.split(' ')
    if int(invalid) == 1 or int(blur) > FLAGS.blur_level:
        return True, None
    return False, (x1, y1, w, h)


def parse_image_info(file):
    while True:
        image = file.readline().rstrip(os.linesep)
        if image is '':
            return None

        bbx_lines = int(file.readline().rstrip(os.linesep))
        # There are some damaged data, eg '0--Parade/0_Parade_Parade_0_452.jpg' in training set,
        # which not follow the bounding box rule, just skip it.
        if bbx_lines == 0:
            file.readline()
            continue

        bboxes = []
        for i in range(bbx_lines):
            bypassed, bbx_info = parse_bbx_info(file)
            if not bypassed:
                bboxes.append(bbx_info)

        if not len(bboxes) == 0 and len(bboxes) <= FLAGS.max_boxes:
            return image, bboxes


def main(_argv):
    output_dir = FLAGS.data_path
    label_files = ['wider_face_train_bbx_gt.txt', 'wider_face_val_bbx_gt.txt']
    output_files = ['wider_face_train_bbx_gt_for_yolo.txt', 'wider_face_val_bbx_gt_for_yolo.txt']
    image_paths = [os.path.join('WIDER_train', 'images'), os.path.join('WIDER_val', 'images')]

    for label_file, output_file, image_path in zip(label_files, output_files, image_paths):
        output = open(os.path.join(output_dir, output_file), 'w+')
        with open(os.path.join(FLAGS.data_path, label_file)) as f:
            while True:
                image_info = parse_image_info(f)
                if image_info is None:
                    break

                logging.info('Processing %s: %s' % (image_path, image_info[0]))
                image_shape = get_image_shape(os.path.join(FLAGS.data_path, image_path, image_info[0]))
                if image_shape is None:
                    continue
                output.write(build_yolo_label(image_info, image_shape))
        output.close()
    return 0


if __name__ == '__main__':
    app.run(main)
