from face_client.state import State

import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont


def frame_gap(frame_a, frame_b):
    if frame_a is None or frame_b is None:
       return 0
    frame_a_blur = cv2.blur(frame_a, (5, 5))
    frame_b_blur = cv2.blur(frame_b, (5, 5))
    frame_delta = cv2.absdiff(frame_a_blur, frame_b_blur)
    thresh = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(thresh, 5, 1, cv2.THRESH_BINARY)
    above_threshold_pixels_portion = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1])
    return above_threshold_pixels_portion


def to_api_data(image):
  _, buffer = cv2.imencode('.png', image)
  return buffer.tostring()


def draw_bbox(frame, bbox):
    if not len(bbox) == 4:
        return frame
    x1, y1, x2, y2 = bbox
    rows, columns, channels = frame.shape
    frame = cv2.rectangle(frame, (int(x1 * columns), int(y1 * rows)), (int(x2 * columns), int(y2 * rows)), (200, 200, 200), 2)
    return frame


def draw_name(frame, name):
    rows, columns, channels = frame.shape
    if name is None:
        text = 'NOT RECOGNIZED'
        color = (0, 0, 255)
        frame = cv2.putText(frame, text, (int(rows * 0.1), int(columns * 0.1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2)
        return frame
    else:
        text = name
        color = (0, 255, 0)
        # Opencv cannot write Chinese, so have to convert to pillow instead.
        # frame = cv2.putText(frame, text, (int(rows * 0.1), int(columns * 0.1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2) 
        frame = Image.fromarray(frame)
        font = ImageFont.truetype('Songti.ttc', 50)
        draw = ImageDraw.Draw(frame)
        draw.text((int(columns * 0.1), int(rows * 0.1)),  text, font=font, fill = color)
        frame = np.array(frame)
        return frame

   
def draw_state(frame, state):
    if state == State.FACE_RECOGNITION_COMPLETED:
        # Do not draw state, draw_name() will use that position.
        return frame
    elif state == State.IDLE:
        text = 'IDLE'
        color = (128, 128, 128)
    else:
        text = 'RECOGNIZING'
        color = (44, 125, 222)
    rows, columns, channels = frame.shape
    frame = cv2.putText(frame, text, (int(rows * 0.1), int(columns * 0.1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2)
    return frame
      

def crop(image, bounding_box):
    x1, y1, x2, y2 = bounding_box
    rows, columns, channels = image.shape
    return image[int(rows * y1): int(rows * y2), int(columns * x1): int(columns * x2), :]


def resize(image, size):
    return cv2.resize(image, (size, size), interpolation = cv2.INTER_LINEAR)
