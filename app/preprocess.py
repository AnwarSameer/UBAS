import cv2, numpy as np
from typing import Tuple, Optional
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def _to_bgr(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("Invalid image data")
    return im

def _face_mesh_landmarks_both_eyes(img_bgr: np.ndarray) -> Optional[dict]:
    # Returns centers and bounds for both eyes using Face Mesh indices
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark

        LEFT_EYE_IDXS  = [33, 133, 159, 145, 246, 161, 163, 7]
        RIGHT_EYE_IDXS = [362, 263, 386, 374, 466, 388, 390, 249]

        def pts(idx_list):
            pts = np.array([(lm[i].x*w, lm[i].y*h) for i in idx_list], dtype=np.float32)
            cx, cy = pts[:,0].mean(), pts[:,1].mean()
            x0, y0 = pts[:,0].min(), pts[:,1].min()
            x1, y1 = pts[:,0].max(), pts[:,1].max()
            return {"cx": cx, "cy": cy, "bbox": (x0, y0, x1, y1), "poly": pts}

        return {"L": pts(LEFT_EYE_IDXS), "R": pts(RIGHT_EYE_IDXS)}

def _expand_bbox(bbox, scale: float, w: int, h: int):
    x0,y0,x1,y1 = bbox
    cx, cy = (x0+x1)/2, (y0+y1)/2
    bw, bh = (x1-x0)*scale, (y1-y0)*scale
    x0n, x1n = int(max(0, cx-bw/2)), int(min(w, cx+bw/2))
    y0n, y1n = int(max(0, cy-bh/2)), int(min(h, cy+bh/2))
    return x0n, y0n, x1n, y1n

def crop_front_both_eyes(img_bgr: np.ndarray, target=(640, 640)) -> Tuple[np.ndarray, dict]:
    h, w = img_bgr.shape[:2]
    info = _face_mesh_landmarks_both_eyes(img_bgr)
    if info is None:
        side = min(h, w)
        x0 = (w - side)//2; y0 = (h - side)//2
        crop = img_bgr[y0:y0+side, x0:x0+side]
    else:
        L, R = info["L"]["bbox"], info["R"]["bbox"]
        x0 = int(min(L[0], R[0])); y0 = int(min(L[1], R[1]))
        x1 = int(max(L[2], R[2])); y1 = int(max(L[3], R[3]))
        x0,y0,x1,y1 = _expand_bbox((x0,y0,x1,y1), scale=3.2, w=w, h=h)
        crop = img_bgr[y0:y1, x0:x1]

    crop_res = cv2.resize(crop, target, interpolation=cv2.INTER_AREA)
    return crop_res, {"crop_xyxy": (x0, y0, x1, y1), "orig_hw": (h, w), "target": target}

def crop_side_single_eye(img_bgr: np.ndarray, target=(640, 640)) -> Tuple[np.ndarray, dict]:
    h, w = img_bgr.shape[:2]
    info = _face_mesh_landmarks_both_eyes(img_bgr)
    if info is None:
        side = min(h, w)
        x0 = (w - side)//2; y0 = (h - side)//2
        crop = img_bgr[y0:y0+side, x0:x0+side]
        crop_res = cv2.resize(crop, target, interpolation=cv2.INTER_AREA)
        return crop_res, {"crop_xyxy": (x0, y0, x0+side, y0+side), "orig_hw": (h, w), "target": target}

    areas = {}
    for k in ("L","R"):
        x0,y0,x1,y1 = info[k]["bbox"]
        areas[k] = (x1-x0)*(y1-y0)
    visible = "L" if areas["L"] >= areas["R"] else "R"
    x0,y0,x1,y1 = _expand_bbox(info[visible]["bbox"], scale=4.0, w=w, h=h)
    crop = img_bgr[y0:y1, x0:x1]
    crop_res = cv2.resize(crop, target, interpolation=cv2.INTER_AREA)
    return crop_res, {"eye": visible, "crop_xyxy": (x0, y0, x1, y1), "orig_hw": (h, w), "target": target}

def preprocess_any(img_bytes: bytes, view: str):
    bgr = _to_bgr(img_bytes)
    if view == "front":
        return crop_front_both_eyes(bgr, target=(640,640))
    else:
        return crop_side_single_eye(bgr, target=(640,640))
