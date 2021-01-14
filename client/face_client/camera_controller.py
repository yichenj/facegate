import cv2


class CameraController:
    def __init__(self):
        self.__camera = cv2.VideoCapture(0)

    def retreive(self):
        return self.__camera.read()

    def close(self):
        self.__camera.release()
