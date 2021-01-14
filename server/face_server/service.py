import json

from face_server import app
from face_server import db
from face_server import np_utils
from face_server import rpc

from flask import current_app
from flask import request
from PIL import Image

FACE_DETECTION_MODEL_INPUT_SIZE = (416, 416)
FACE_RECOGNITION_MODEL_INPUT_SIZE = (112, 112)

@app.route('/detect/<uuid>', methods=['POST'])
def detect(uuid):
    image = Image.open(request.stream)
    if not image.size == FACE_DETECTION_MODEL_INPUT_SIZE:
        image = image.resize(FACE_DETECTION_MODEL_INPUT_SIZE, resample=Image.BILINEAR)
    num_of_objects, boxes, confidence = rpc.face_detect(np_utils.to_array(image))

    score_threshold = current_app.config['FACE_DETECTION_SCORE_THRESHOLD']
    area_threshold = current_app.config['FACE_DETECTION_AREA_THRESHOLD']
    instances = []
    for i in range(num_of_objects):
        if confidence[i] < score_threshold:
            continue
        if (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) < area_threshold:
            continue

        bounding_box = [float(boxes[i][x]) for x in range(4)]
        score = float(confidence[i])
        instances.append({'bounding_box': bounding_box, 'score': score})

    return {'uuid': uuid, 'instances': instances}


@app.route('/recognize/<uuid>', methods=['POST'])
def recognize(uuid):
    image = Image.open(request.stream)
    if not image.size == FACE_RECOGNITION_MODEL_INPUT_SIZE:
        image = image.resize(FACE_RECOGNITION_MODEL_INPUT_SIZE, resample=Image.BILINEAR)
    embedding = rpc.get_face_embedding(np_utils.to_array(image))
    embedding = np_utils.l2_normalize(embedding)
    
    connection = db.get_connection()
    cursor = connection.cursor()
    offset = 0
    name = None
    distance = 0
    threshold = current_app.config['FACE_RECOGNITION_THRESHOLD']
    while True:
        cursor.execute('SELECT * FROM faces LIMIT ? OFFSET ?', (100, offset))
        rows = cursor.fetchall()
        if len(rows) == 0:
            break       

        for row in rows:
            d = np_utils.distance(row[2], embedding)
            print(row[1], d)
            if d > threshold:
                name = row[1]
                distance = float(d)
                break

        offset = offset + 100

    return {'uuid': uuid, 'name': name, 'distance': distance}


@app.route('/register', methods=['POST'])
def register():
    meta = json.loads(request.form.get('metadata'))
    name = meta.pop('name', None)
    if not name:
        return {'msg': 'name field cannot be empty'}, 400

    image = Image.open(request.files.get('image').stream)
    if not image.size == FACE_RECOGNITION_MODEL_INPUT_SIZE:
        image = image.resize(FACE_RECOGNITION_MODEL_INPUT_SIZE, resample=Image.BILINEAR)
    embedding = rpc.get_face_embedding(np_utils.to_array(image))
    embedding = np_utils.l2_normalize(embedding)

    connection = db.get_connection()
    connection.execute('INSERT INTO faces VALUES (?, ?, ?)', (None, name, embedding))
    connection.commit()
    return {'msg': 'success'}
