import sys
import logging

from face_client import face_client
from face_client.api_proxy import FaceApiProxy
from face_client.camera_controller import CameraController
from face_client.image_displayer import ImageDisplayer


def main(argv):
    client = face_client.FaceClient(CameraController(), ImageDisplayer(), FaceApiProxy())
    client.start()


def run_main():  # pylint: disable=invalid-name
    try:
        sys.exit(main(sys.argv))
    except Exception as e:
        logging.exception('face client crashed...')
        sys.exit(1)
