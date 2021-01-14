import cv2


class ImageDisplayer:
    def __init__(self):
        pass

    def play(self, frame):
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(100)
        # Press ESC to quit
        if key == 27:
            return True
        return False

    def close(self):
        cv2.destroyAllWindows()
