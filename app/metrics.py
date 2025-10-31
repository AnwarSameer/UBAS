import numpy as np
from typing import List, Tuple, Dict
from .schemas import LandmarkSet, FrontMetrics, SideFeatures, SideMetrics, Calibration
import math

def _interp_y_on_polyline(poly: List[Tuple[float,float]], x: float) -> float:
    pts = np.array(poly)
    idx = np.argmin(np.abs(pts[:,0] - x))
    return float(pts[idx,1])

def _dist_vertical(poly: List[Tuple[float,float]], x: float, y_from: float) -> float:
    y_on_poly = _interp_y_on_polyline(poly, x)
    return y_on_poly - y_from

def _column_x(mc: Tuple[float,float], lc: Tuple[float,float], frac: float) -> float:
    a, b = np.array(mc), np.array(lc)
    return float(a[0] + frac*(b[0]-a[0]))

def _area_ratio(region_mask_px: int, iris_area_px: float) -> float:
    return region_mask_px / iris_area_px if iris_area_px>0 else 0.0

def _scale_mm_per_px(calib: Calibration, iris_radius_px: float) -> float:
    if calib.mode == "sticker" and calib.sticker_px and calib.sticker_diam_mm:
        return calib.sticker_diam_mm / calib.sticker_px
    return calib.iris_diam_mm / (2*iris_radius_px)

def front_metrics(lm_L: LandmarkSet, lm_R: LandmarkSet,
                  calib: Calibration,
                  lateral_fold_area_px_L: int=0, lateral_fold_area_px_R: int=0):
    def per_eye(lm: LandmarkSet) -> Dict[str, float]:
        Cx, Cy, r = lm.iris_center[0], lm.iris_center[1], lm.iris_radius
        mrd1 = ( _dist_vertical(lm.upper_lid, Cx, Cy) * -1 ) / r
        mrd2 = ( _dist_vertical(lm.lower_lid, Cx, Cy) ) / r
        pfh = mrd1 + mrd2
        x_med = _column_x(lm.medial_canthus, lm.lateral_canthus, 0.35)
        x_lat = _column_x(lm.medial_canthus, lm.lateral_canthus, 0.70)
        def tps_at(x):
            d = (_interp_y_on_polyline(lm.lash_line, x) - _interp_y_on_polyline(lm.crease_line, x)) / r
            return -d
        tps_mid = (_interp_y_on_polyline(lm.lash_line, Cx) - _interp_y_on_polyline(lm.crease_line, Cx)) / r * -1
        return dict(
            mrd1=mrd1, mrd2=mrd2, pfh=pfh,
            tps_mid=tps_mid, tps_med=tps_at(x_med), tps_lat=tps_at(x_lat),
            bpd=( (_interp_y_on_polyline(lm.brow_curve, Cx) - Cy) * -1 ) / r
        )

    L, R = per_eye(lm_L), per_eye(lm_R)
    v = np.array(lm_L.lateral_canthus) - np.array(lm_L.medial_canthus)
    canthal_tilt = float(np.degrees(np.arctan2(v[1], v[0])))

    iris_area_L = math.pi * (lm_L.iris_radius**2)
    iris_area_R = math.pi * (lm_R.iris_radius**2)
    lhi_L = _area_ratio(lateral_fold_area_px_L, iris_area_L)
    lhi_R = _area_ratio(lateral_fold_area_px_R, iris_area_R)

    mm_per_px = _scale_mm_per_px(calib, (lm_L.iris_radius + lm_R.iris_radius)/2)

    fm = FrontMetrics(
        mrd1_L=L["mrd1"], mrd2_L=L["mrd2"], pfh_L=L["pfh"],
        tps_mid_L=L["tps_mid"], tps_med_L=L["tps_med"], tps_lat_L=L["tps_lat"],
        ech_cols_L=[],  # could sample 6 columns
        bpd_L=L["bpd"],
        mrd1_R=R["mrd1"], mrd2_R=R["mrd2"], pfh_R=R["pfh"],
        tps_mid_R=R["tps_mid"], tps_med_R=R["tps_med"], tps_lat_R=R["tps_lat"],
        ech_cols_R=[],
        bpd_R=R["bpd"],
        canthal_tilt_deg=canthal_tilt,
        lat_hooding_idx_L=lhi_L, lat_hooding_idx_R=lhi_R,
        ci={}
    )
    return fm, mm_per_px

def side_metrics(sf: SideFeatures) -> SideMetrics:
    r = sf.iris_radius
    skin = np.array(sf.skin_above_crease); crease = np.array(sf.crease_line)
    n = min(len(skin), len(crease))
    area = np.trapz((skin[:n,1] - crease[:n,1]), dx=1.0)
    sci = (area / (r**2)) if r>0 else 0.0
    apex_idx = np.argmin(skin[:,1])
    brow_apex = skin[apex_idx]
    bgl = (sf.corneal_apex[1] - brow_apex[1]) / r if r>0 else 0.0
    return SideMetrics(
        sulcus_concavity_idx=float(sci),
        brow_globe_vector=float(bgl),
        lash_vector_angle_delta_deg=0.0,
        ci={}
    )
