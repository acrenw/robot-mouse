"""
cat detection using yolov8s (nano missed the cat too much, so we're trading speed for accuracy) + bytetrack

persist=True keeps bytetrack's kalman filter running between frames so it
can predict where the cat is even when yolo misses a frame

usage:
    `python run/detect.py data/videos/pounce-5.mp4`

TODO: fine-tune yolov8s on dads cat footage to improve detection reliability
TODO: try lower conf threshold (0.05) in bad lighting conditions
TODO: wider FOV camera might help so the cat doesn't disappear at frame edges
"""

import cv2
import numpy as np
import sys

try:
    from ultralytics import YOLO
    _model = None

    def _get_model():
        global _model
        if _model is None:
            _model = YOLO('yolov8s.pt') # downloads ~22mb on first run
        return _model

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[detect] ultralytics not found, using stub obs")


FRAME_W = 640
FRAME_H = 480
CAT_CLASS = 15 # coco class 15 = cat


def get_cat_state(frame, prev_bbox=None, imgsz=320):
    """
    returns:
        state: np.ndarray (5,) [dist_proxy, angle, visible, vx, vy]
        cur_bbox: (x1, y1, x2, y2) or None

    dist_proxy: 0 = cat filling frame (very close), 1 = tiny or not found
    angle: -1 = far left, 0 = center, +1 = far right (positive = cat right of center)
    visible: 1.0 if yolo returned a box this frame, else 0.0
    vx, vy: normalized pixel velocity of bbox center between frames (* 10)

    imgsz=320 for pi (faster), 640 for laptop (more accurate)
    """
    if not YOLO_AVAILABLE:
        return _stub_state(), None

    model = _get_model()
    # enabling persist allows ultralytics to track a box to the same cat from kalman filter predictions (box.id continuity) not really being used rn tho
    results = model.track(frame, classes=[CAT_CLASS], persist=True, verbose=False, imgsz=imgsz, conf=0.10)

    # get most prominent cat in frame
    best = None
    best_area = 0
    
    for r in results:
        for box in r.boxes: # for cat in all cats detected
            x1, y1, x2, y2 = box.xyxy[0].tolist() # theres only one element, a (1, 4) tensor in here
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)

    if best is None:
        # no cat AKA cat is maximally far
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), None

    x1, y1, x2, y2 = best
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    # large bbox = cat is close to camera
    bbox_area_norm = (w * h) / (FRAME_W * FRAME_H)  # what fraction of the frame the cat takes up (0 to 1)
    
    # *4 so that a cat taking up 25% of the frame reads as dist=0 (very close)
    # without *4 you'd need the cat to fill the whole frame to get dist=0, which never happens
    dist_proxy = float(np.clip(1.0 - bbox_area_norm * 4, 0.0, 1.0))

    # horizontal angle from frame center, -1 to +1
    # (cx - center) / (center) gives -1 at left edge, 0 at center, +1 at right edge
    angle = float(np.clip((cx - FRAME_W / 2.0) / (FRAME_W / 2.0), -1.0, 1.0))

    vx, vy = 0.0, 0.0
    if prev_bbox is not None:
        px1, py1, px2, py2 = prev_bbox
        # pixel displacement of bbox center between frames, normalized by frame size
        # *10 to scale it up so it's a useful signal (raw pixel/frame would be tiny)
        vx = ((cx - (px1 + px2) / 2.0) / FRAME_W) * 10.0
        vy = ((cy - (py1 + py2) / 2.0) / FRAME_H) * 10.0

    state = np.array([dist_proxy, angle, 1.0, vx, vy], dtype=np.float32)

    return state, (x1, y1, x2, y2)


def _stub_state():
    # stub so everything is neutral state
    return np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def draw_debug(frame, cat_state, bbox):
    """
    draw bounding box and state overlay
    stale=True tints everything orange
    """
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        colour = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, "cat", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    dist, angle, vis, vx, vy = cat_state
    label = f"d={dist:.2f} ang={angle:.2f} vis={vis:.0f}"
    colour = (255, 255, 0)

    cv2.putText(frame, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    return frame


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"can't open: {src}")
        # exit code 1 for smth went wrong
        sys.exit(1)

    print("running detection (yolov8s + bytetrack), press q to quit")

    prev_bbox = None

    while True:
        ret, frame = cap.read() # frame's shape: (height, width, 3), 3 is colour in B, G, R (opencv convention)
        if not ret: # if no frame was grabbed
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        state, bbox = get_cat_state(frame, prev_bbox)
        prev_bbox = bbox

        frame = draw_debug(frame, state, bbox)

        print(f"  state: {state}")
        cv2.imshow("detect", frame)

        # Q key detection
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
