import numpy as np
from typing import List, Tuple
from .schemas import LandmarkSet, QCResult

def _primary_gaze(landmarks: LandmarkSet, can_thresh_deg=3.0) -> bool:
    mc, lc = np.array(landmarks.medial_canthus), np.array(landmarks.lateral_canthus)
    horiz = lc - mc
    angle = np.degrees(np.arctan2(horiz[1], horiz[0]))
    return abs(angle) <= can_thresh_deg

def _head_roll_ok(face_pose_deg: float, limit=3.0) -> bool:
    return abs(face_pose_deg) <= limit

def run_qc(front_gray, side_gray, front_lm: LandmarkSet, face_roll_deg: float,
           min_res=(480, 480)) -> QCResult:
    reasons: List[str] = []
    h, w = front_gray.shape[:2]
    if h < min_res[0] or w < min_res[1]:
        reasons.append("Low resolution: need ≥ 480×480.")
    if not _primary_gaze(front_lm):
        reasons.append("Eye not in primary gaze (canthal line not horizontal).")
    if not _head_roll_ok(face_roll_deg):
        reasons.append("Head tilt > 3°.")
    return QCResult(passed=(len(reasons)==0), reasons=reasons)
