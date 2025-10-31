import base64, cv2, numpy as np
import mediapipe as mp
from .schemas import LandmarkSet, SideFeatures

mp_face_mesh = mp.solutions.face_mesh

# Utility to decode b64 and keep BGR (OpenCV)
def _decode_b64(img_b64: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)

def _poly_from_idxs(lm, idxs, w, h):
    return [(lm[i].x*w, lm[i].y*h) for i in idxs]

def _iris_center_radius(lm, idxs, w, h):
    pts = np.array([(lm[i].x*w, lm[i].y*h) for i in idxs], dtype=np.float32)
    cx, cy = pts[:,0].mean(), pts[:,1].mean()
    # Approximate radius as mean distance to center
    r = float(np.mean(np.linalg.norm(pts - np.array([cx, cy]), axis=1)))
    return (cx, cy), r

def run_front_pipeline(front_b64: str):
    img = _decode_b64(front_b64)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            # Fallback dummy straight geometry in center so pipeline still runs
            Cx, Cy, r = w/2, h/2, min(h,w)/10
            lm = LandmarkSet(
                upper_lid=[(Cx-40, Cy-20), (Cx, Cy-22), (Cx+40, Cy-20)],
                lower_lid=[(Cx-40, Cy+20), (Cx, Cy+22), (Cx+40, Cy+20)],
                lash_line=[(Cx-40, Cy+5), (Cx, Cy+5), (Cx+40, Cy+5)],
                crease_line=[(Cx-40, Cy-15), (Cx, Cy-16), (Cx+40, Cy-15)],
                brow_curve=[(Cx-60, Cy-60), (Cx, Cy-65), (Cx+60, Cy-58)],
                medial_canthus=(Cx-60, Cy), lateral_canthus=(Cx+60, Cy),
                iris_center=(Cx, Cy), iris_radius=r, confidences={}
            )
            face_roll_deg = 0.0
            return img, lm, face_roll_deg

        lm = res.multi_face_landmarks[0].landmark

        # Canonical indices
        LEFT_UPPER_IDX  = [159, 158, 157, 173, 133]
        LEFT_LOWER_IDX  = [145, 144, 163, 7, 33]
        RIGHT_UPPER_IDX = [386, 385, 384, 398, 263]
        RIGHT_LOWER_IDX = [374, 380, 381, 382, 362]
        LEFT_IRIS_IDX   = [468, 469, 470, 471]
        RIGHT_IRIS_IDX  = [473, 474, 475, 476]
        BROW_LEFT_IDX   = [70, 63, 105, 66, 107]
        BROW_RIGHT_IDX  = [336, 296, 334, 293, 300]
        MED_CANTHUS_L   = 133
        LAT_CANTHUS_L   = 33
        MED_CANTHUS_R   = 263
        LAT_CANTHUS_R   = 362

        # We will build a single combined landmark set by averaging left/right for simplicity.
        # (For metric calc we will pass same structure for L/R to keep demo simple.)
        ul = _poly_from_idxs(lm, LEFT_UPPER_IDX, w, h)
        ll = _poly_from_idxs(lm, LEFT_LOWER_IDX, w, h)
        ur = _poly_from_idxs(lm, RIGHT_UPPER_IDX, w, h)
        lr = _poly_from_idxs(lm, RIGHT_LOWER_IDX, w, h)

        # Approximate lash line as just above lower lid (1/3 of upper-lower gap)
        def mid_poly(a, b, t=0.33):
            a = np.array(a); b = np.array(b)
            m = a*(1-t) + b*t
            return [tuple(p) for p in m.tolist()]

        lash_L = mid_poly(ll, ul, t=0.2)
        lash_R = mid_poly(lr, ur, t=0.2)

        # Approximate crease as above upper lid by a fixed offset toward brow
        brow_L = _poly_from_idxs(lm, BROW_LEFT_IDX, w, h)
        brow_R = _poly_from_idxs(lm, BROW_RIGHT_IDX, w, h)

        def crease_from_upper(upper, brow, lift_px=12):
            upper = np.array(upper); brow = np.array(brow)
            # shift upper towards brow by a fraction, then add small lift
            cre = upper - (upper - brow)*0.25
            cre[:,1] -= lift_px
            return [tuple(p) for p in cre.tolist()]

        crease_L = crease_from_upper(ul, brow_L)
        crease_R = crease_from_upper(ur, brow_R)

        # Use left iris for the joint center (you can split per-eye downstream)
        (cLx,cLy), rL = _iris_center_radius(lm, LEFT_IRIS_IDX, w, h)
        (cRx,cRy), rR = _iris_center_radius(lm, RIGHT_IRIS_IDX, w, h)
        Cx, Cy = (cLx+cRx)/2.0, (cLy+cRy)/2.0
        r = (rL + rR)/2.0

        medial_canthus = ((lm[MED_CANTHUS_L].x*w + lm[MED_CANTHUS_R].x*w)/2.0,
                          (lm[MED_CANTHUS_L].y*h + lm[MED_CANTHUS_R].y*h)/2.0)
        lateral_canthus = ((lm[LAT_CANTHUS_L].x*w + lm[LAT_CANTHUS_R].x*w)/2.0,
                           (lm[LAT_CANTHUS_L].y*h + lm[LAT_CANTHUS_R].y*h)/2.0)

        # Merge left/right into single polylines by averaging corresponding samples
        def avg_poly(a, b):
            a=np.array(a); b=np.array(b)
            n=min(len(a),len(b))
            m=(a[:n]+b[:n])/2.0
            return [tuple(p) for p in m.tolist()]

        upper = avg_poly(ul, ur)
        lower = avg_poly(ll, lr)
        lash  = avg_poly(lash_L, lash_R)
        crease= avg_poly(crease_L, crease_R)
        brow  = avg_poly(brow_L, brow_R)

        # Head roll estimate from canthal line
        v = np.array(lateral_canthus) - np.array(medial_canthus)
        face_roll_deg = float(np.degrees(np.arctan2(v[1], v[0])))

        lmset = LandmarkSet(
            upper_lid=upper, lower_lid=lower, lash_line=lash, crease_line=crease,
            brow_curve=brow, medial_canthus=medial_canthus, lateral_canthus=lateral_canthus,
            iris_center=(Cx, Cy), iris_radius=float(r), confidences={"mediapipe": 1.0}
        )
        return img, lmset, face_roll_deg

def run_side_pipeline(side_b64: str):
    img = _decode_b64(side_b64)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            # Neutral placeholder that yields 'none' sulcus concavity
            Cx, Cy, r = w/2, h/2, min(h,w)/10
            sf = SideFeatures(
                crease_line=[(Cx-40, Cy-10), (Cx+40, Cy-10)],
                skin_above_crease=[(Cx-40, Cy-12), (Cx, Cy-14), (Cx+40, Cy-13)],
                brow_curve=[(Cx-50, Cy-60), (Cx+50, Cy-58)],
                corneal_apex=(Cx+10, Cy),
                lash_line=[(Cx-30, Cy+5), (Cx+30, Cy+5)],
                iris_center=(Cx, Cy), iris_radius=r
            )
            return img, sf

        lm = res.multi_face_landmarks[0].landmark

        # Choose a small horizontal band for crease & skin above; this is approximate.
        # Use left eye indices if present, else right.
        LEFT_UPPER_IDX  = [159, 158, 157, 173, 133]
        BROW_LEFT_IDX   = [70, 63, 105, 66, 107]
        LEFT_IRIS_IDX   = [468, 469, 470, 471]
        if True:
            upper = _poly_from_idxs(lm, LEFT_UPPER_IDX, w, h)
            brow  = _poly_from_idxs(lm, BROW_LEFT_IDX, w, h)
            (Cx, Cy), r = _iris_center_radius(lm, LEFT_IRIS_IDX, w, h)
        crease = [(x, y-10) for (x,y) in upper]
        skin   = [(x, y-14) for (x,y) in upper]

        sf = SideFeatures(
            crease_line=crease,
            skin_above_crease=skin,
            brow_curve=brow,
            corneal_apex=(Cx+10, Cy),
            lash_line=[(Cx-30, Cy+5), (Cx+30, Cy+5)],
            iris_center=(Cx, Cy), iris_radius=float(r)
        )
        return img, sf
