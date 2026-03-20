import cv2


def open_camera(camera_index: int = 0, width: int = 1280, height: int = 720):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    return cap