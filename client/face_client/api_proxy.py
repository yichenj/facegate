from concurrent import futures
import functools
import time
import threading
import requests


class FaceApiProxy:
    def __init__(self, server_url='http://127.0.0.1:8000', max_workers=8):
        self.__pool = futures.ThreadPoolExecutor(max_workers=max_workers)
        self.__http_sesstion = requests.Session()
        self.__server_url = server_url
        self.__callback_queue = None

    def set_callback_queue(self, queue):
        self.__callback_queue = queue

    def submit(self, task, uuid, callback):
        running_task = self.__pool.submit(task)
        running_task.arg = uuid
        if self.__callback_queue is not None and callback is not None:
            running_task.add_done_callback(lambda t: self.__api_callback_wrapper(t, callback))

    def face_detect(self, uuid, image, callback):
        def face_detection_request():
#            time.sleep(5)
#            return {'uuid': str(uuid), 'instances': [{'bounding_box': [0.5, 0.5, 0.75, 0.75], 'score': 0.9}]}
            response = self.__http_sesstion.post(self.__server_url + '/detect/' + str(uuid), data=image, timeout=5)
            return response.json()
        self.submit(face_detection_request, uuid, callback)

    def face_recognize(self, uuid, image, callback):
        def face_recognition_request():
#            time.sleep(1)
#            return {'uuid': str(uuid), 'name': '蒋逸尘', 'distance': 0}
            response = self.__http_sesstion.post(self.__server_url + '/recognize/' + str(uuid), data=image, timeout=5)
            return response.json()
        self.submit(face_recognition_request, uuid, callback)

    def __api_callback_wrapper(self, running_task, callback):
        if running_task.cancelled():
            print('{}: canceled'.format(running_task.arg))
            error = None
            result = {}
        elif running_task.done():
            error = running_task.exception()
            if error:
                print('{}: error returned: {}'.format(
                    running_task.arg, error))
                result = {}
            else:
                result = running_task.result()
                print('{}: value returned: {}'.format(
                    running_task.arg, result))

        self.__callback_queue.put(functools.partial(callback, running_task.arg, error, result))
