import uuid

class RecognitionContext:
    def __init__(self, src_image):
        self.__uuid = uuid.uuid1()
        self.__src_image = src_image
        self.__detection_result = None
        self.__recognition_result = None

    def uuid(self):
        return self.__uuid

    def get_detection_result(self):
        return self.__detection_result

    def set_detection_result(self, result):
        self.__detection_result = result

    def get_recognition_result(self):
        return self.__recognition_result

    def set_recognition_result(self, result):
        self.__recognition_result = result

    def get_image(self):
        return self.__src_image
