import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

_channel = None
_stub = None


def init(app):
    global _channel, _stub
    _channel = grpc.insecure_channel(app.config['RPC_URI'])
    _stub = prediction_service_pb2_grpc.PredictionServiceStub(_channel)


def get_face_embedding(image):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'arcface'
    request.model_spec.signature_name = 'serving_default'

    image = np.expand_dims(image, axis=0)
    request.inputs['input_image'].CopyFrom(tf.make_tensor_proto(image))

    response = _stub.Predict(request, timeout=5.0)
    return tf.make_ndarray(response.outputs['OutputLayer'])[0]


def face_detect(image):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'yolov3'
    request.model_spec.signature_name = 'serving_default'

    image = np.expand_dims(image, axis=0)
    request.inputs['input'].CopyFrom(tf.make_tensor_proto(image))

    response = _stub.Predict(request, timeout=5.0)
    boxes = tf.make_ndarray(response.outputs['yolo_nms'])[0]
    confidence = tf.make_ndarray(response.outputs['yolo_nms_1'])[0]
    class_id = tf.make_ndarray(response.outputs['yolo_nms_2'])[0]
    num_of_objects = tf.make_ndarray(response.outputs['yolo_nms_3'])[0]

    # We have only one class of face.
    for i in range(num_of_objects):
        assert class_id[i] == 0
    return num_of_objects, boxes[:num_of_objects], confidence[:num_of_objects]
