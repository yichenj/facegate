import queue

from face_client import image_utils
from face_client.state import State
from face_client.recognition_context import RecognitionContext


class FaceClient:
    def __init__(self, camera_controller, image_displayer, api_proxy):
        self.__camera_controller = camera_controller
        self.__image_displayer = image_displayer
        self.__api_proxy = api_proxy

        self.__api_proxy_callbacks = queue.Queue()
        self.__api_proxy.set_callback_queue(self.__api_proxy_callbacks)

        self.__state = State.AWAKING
        self.__recognition_context = None

    def start(self):
        last_frame, current_frame = None, None
        while True:
            self.__run_callbacks()

            success, frame = self.__camera_controller.retreive()
            if success:
                last_frame, current_frame = current_frame, frame
                frame_gap = image_utils.frame_gap(last_frame, current_frame)
                print('Current state: %s, frame gap: %s' % (self.__state, frame_gap))
                if frame_gap > 0.5:
                    self.__state = State.AWAKING

                frame = self.__do_recognition_routine(current_frame)
            quit = self.__image_displayer.play(frame)
            if quit:
                break

        self.__camera_controller.close()
        self.__image_displayer.close()

    def __run_callbacks(self):
      try:
        while True:
          callback = self.__api_proxy_callbacks.get_nowait()
          callback()
      except queue.Empty:
          return
 
    def __on_face_detected(self, recognition_uuid, error, result):
        if (self.__recognition_context == None or
            not self.__recognition_context.uuid() == recognition_uuid):
            return
        if error is not None:
            self.__state = State.AWAKING
            return
        instances = result.pop('instances', None)
        if len(instances) <= 0:
            self.__state = State.IDLE
            return
        self.__recognition_context.set_detection_result(instances[0].pop('bounding_box', []))
        self.__state = State.FACE_DETECTION_COMPLETED

    def __on_face_recoginized(self, recognition_uuid, error, result):
        if (self.__recognition_context == None or
            not self.__recognition_context.uuid() == recognition_uuid):
            return

        if error is not None:
            self.__state = State.FACE_DETECTION_COMPLETED
            return

        name = result.pop('name', None)
        self.__recognition_context.set_recognition_result(name)
        self.__state = State.FACE_RECOGNITION_COMPLETED


    def __do_recognition_routine(self, current_frame):
        state = self.__state
        if state == State.IDLE:
            pass
        elif state == State.AWAKING:
            detect_image = image_utils.resize(current_frame.copy(), 416)
            self.__recognition_context = RecognitionContext(detect_image)
            print('New recognition process %s' % self.__recognition_context.uuid())
            self.__api_proxy.face_detect(self.__recognition_context.uuid(), image_utils.to_api_data(detect_image), self.__on_face_detected)
            self.__state = State.DO_FACE_DETECTION
        elif state == State.DO_FACE_DETECTION:
            pass
        elif state == State.FACE_DETECTION_COMPLETED:
            bbox = self.__recognition_context.get_detection_result()
            cropped_image = image_utils.crop(self.__recognition_context.get_image(), bbox)
            recognize_image = image_utils.resize(cropped_image, 112)
            self.__api_proxy.face_recognize(self.__recognition_context.uuid(), image_utils.to_api_data(recognize_image), self.__on_face_recoginized)
            self.__state = State.DO_FACE_RECOGNITION
            current_frame = image_utils.draw_bbox(current_frame, bbox)
        elif state == State.DO_FACE_RECOGNITION:
            bbox = self.__recognition_context.get_detection_result()
            current_frame = image_utils.draw_bbox(current_frame, bbox)
        elif state == State.FACE_RECOGNITION_COMPLETED:
            bbox = self.__recognition_context.get_detection_result()
            current_frame = image_utils.draw_bbox(current_frame, bbox)
            recognition_result = self.__recognition_context.get_recognition_result()
            current_frame = image_utils.draw_name(current_frame, recognition_result)
        else:
            # Not reached
            assert False

        return image_utils.draw_state(current_frame, self.__state)
