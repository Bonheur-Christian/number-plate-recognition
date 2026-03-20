import cv2
import imutils
import numpy as np


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def detect_plate(frame):
    """
    Returns:
        plate_contour: 4-point contour or None
        debug_frame: frame with drawn candidate
    """
    debug_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = cv2.Canny(gray, 30, 200)

    # Force individual text characters to merge into a single solid rectangular blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    candidates = []

    for cnt in contours:
        # Use a rotated bounding box instead of strict 4-point approximation
        # This handles hand-drawn or occluded plates (e.g. held by fingers) where contours are broken.
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        
        if w == 0 or h == 0:
            continue
            
        aspect_ratio = max(w/h, h/w)
        area = w * h

        # rough plate-like shape
        if 2.0 <= aspect_ratio <= 6.5 and area > 1000:
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            candidates.append((area, box))

    # Sort candidates by area descending and take top 5
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:5]
    
    ordered_candidates = []
    for area, quad in candidates:
        cv2.drawContours(debug_frame, [quad], -1, (0, 255, 0), 2)
        pts = np.array(quad, dtype="float32")
        ordered_candidates.append(order_points(pts))

    return ordered_candidates, debug_frame