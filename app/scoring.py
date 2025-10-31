from typing import Dict
import math
from .schemas import FrontMetrics, SideMetrics, UBASScore

def score(front: FrontMetrics, side: SideMetrics=None, preop=None) -> UBASScore:
    rubric: Dict[str, int] = {}

    # 1) TPS gain (mid): using absolute band as proxy if delta not provided
    tps_mid_avg = (front.tps_mid_L + front.tps_mid_R)/2.0
    if tps_mid_avg > 0.35: rubric["TPS gain (mid)"] = 3
    elif 0.25 <= tps_mid_avg <= 0.35: rubric["TPS gain (mid)"] = 2
    elif 0.15 <= tps_mid_avg < 0.25: rubric["TPS gain (mid)"] = 1
    else: rubric["TPS gain (mid)"] = 0

    # 2) TPS balance (medial:lateral)
    med = (front.tps_med_L + front.tps_med_R)/2.0
    lat = (front.tps_lat_L + front.tps_lat_R)/2.0
    ratio = med/lat if abs(lat) > 1e-6 else math.inf
    if 0.8 <= ratio <= 1.2: pts=3
    elif 1.2 < ratio <= 1.3 or 0.77 <= ratio < 0.8: pts=2
    elif 1.3 < ratio <= 1.6 or 0.6 <= ratio < 0.77: pts=1
    else: pts=0
    rubric["TPS balance (M:L)"] = pts

    # 3) MRD1 change proxy
    mrd1_mean = (front.mrd1_L + front.mrd1_R)/2.0
    if mrd1_mean > 0.15: pts=3
    elif 0.10 <= mrd1_mean <= 0.15: pts=2
    elif 0.05 <= mrd1_mean < 0.10: pts=1
    else: pts=0
    rubric["MRD1 change"] = pts

    # 4) PFH band 0.75–0.95
    pfh_mean = (front.pfh_L + front.pfh_R)/2.0
    if 0.75 <= pfh_mean <= 0.95: pts=3
    elif 0.70 <= pfh_mean < 0.75 or 0.95 < pfh_mean <= 1.00: pts=2
    elif 0.60 <= pfh_mean < 0.70 or 1.00 < pfh_mean <= 1.10: pts=1
    else: pts=0
    rubric["PFH band"] = pts

    # 5) Crease symmetry (proxy using TPS mid difference in ID units)
    crease_diff_id = abs(front.tps_mid_L - front.tps_mid_R)
    if crease_diff_id < 0.085: pts=3
    elif crease_diff_id < 0.127: pts=2
    elif crease_diff_id < 0.170: pts=1
    else: pts=0
    rubric["Crease symmetry"] = pts

    # 6) Crease continuity (placeholder 0..1 continuity => we assume near-complete)
    crease_cont_score = 0.9
    if crease_cont_score >= 0.95: pts=3
    elif crease_cont_score >= 0.85: pts=2
    elif crease_cont_score >= 0.70: pts=1
    else: pts=0
    rubric["Crease continuity"] = pts

    # 7) Brow stability (change ~0 if preop not given)
    bpd_change = 0.0
    if abs(bpd_change) <= 0.00: pts=3
    elif abs(bpd_change) < 0.05: pts=2
    elif abs(bpd_change) < 0.10: pts=1
    else: pts=0
    rubric["Brow stability"] = pts

    # Side components (optional)
    if side is None:
        rubric["Sulcus concavity"] = 2  # neutral/mild
        rubric["Brow–globe vector"] = 2
        rubric["Lash vector"] = 2
    else:
        v = side.sulcus_concavity_idx
        if v <= 0.0: pts=3
        elif v <= 0.2: pts=2
        elif v <= 0.5: pts=1
        else: pts=0
        rubric["Sulcus concavity"] = pts

        bgl_change = 0.0
        if abs(bgl_change) <= 0.02: pts=3
        elif abs(bgl_change) <= 0.05: pts=2
        elif abs(bgl_change) <= 0.10: pts=1
        else: pts=0
        rubric["Brow–globe vector"] = pts

        deg = abs(side.lash_vector_angle_delta_deg)
        if deg <= 2: pts=2
        elif deg <= 6: pts=1
        else: pts=0
        rubric["Lash vector"] = pts

    subs = {
        "Front Symmetry": rubric["Crease symmetry"] + rubric["Crease continuity"],
        "Tarsal Show": rubric["TPS gain (mid)"] + rubric["TPS balance (M:L)"],
        "Function (MRD1)": rubric["MRD1 change"],
        "Brow Stability": rubric["Brow–globe vector"] + rubric["Brow stability"],
        "Sulcus Fullness": rubric["Sulcus concavity"] + rubric["Lash vector"]
    }
    total = sum(rubric.values())
    band = ("Excellent" if total>=26 else
            "Good" if total>=21 else
            "Acceptable" if total>=16 else
            "Suboptimal")

    rubric_verbose = {k: {"points": v} for k,v in rubric.items()}
    return UBASScore(total=total, band=band, subscores=subs, rubric=rubric_verbose)
