from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple

Point = Tuple[float, float]

class LandmarkSet(BaseModel):
    # Dense or sampled polylines / key features (front view per eye block)
    upper_lid: List[Point]
    lower_lid: List[Point]
    lash_line: List[Point]
    crease_line: List[Point]
    brow_curve: List[Point]
    medial_canthus: Point
    lateral_canthus: Point
    iris_center: Point
    iris_radius: float  # pixels
    confidences: Optional[Dict[str, float]] = None

class SideFeatures(BaseModel):
    crease_line: List[Point]
    skin_above_crease: List[Point]
    brow_curve: List[Point]
    corneal_apex: Point
    lash_line: List[Point]
    iris_center: Point
    iris_radius: float

class Calibration(BaseModel):
    mode: str = Field(..., regex="^(iris|sticker)$")
    iris_diam_mm: float = 11.8
    sticker_diam_mm: Optional[float] = 10.0
    sticker_px: Optional[float] = None

class QCResult(BaseModel):
    passed: bool
    reasons: List[str] = []

class FrontMetrics(BaseModel):
    mrd1_L: float; mrd1_R: float
    mrd2_L: float; mrd2_R: float
    pfh_L: float;  pfh_R: float
    tps_mid_L: float; tps_mid_R: float
    tps_med_L: float; tps_med_R: float
    tps_lat_L: float; tps_lat_R: float
    ech_cols_L: List[float]; ech_cols_R: List[float]
    bpd_L: float; bpd_R: float
    canthal_tilt_deg: float
    lat_hooding_idx_L: float; lat_hooding_idx_R: float
    ci: Optional[Dict[str, float]] = None

class SideMetrics(BaseModel):
    sulcus_concavity_idx: float
    brow_globe_vector: float
    lash_vector_angle_delta_deg: float
    ci: Optional[Dict[str, float]] = None

class UBASScore(BaseModel):
    total: int
    band: str
    subscores: Dict[str, int]
    rubric: Dict[str, Dict[str, int]]

class AnalyzeResponse(BaseModel):
    qc: QCResult
    front: Optional[FrontMetrics] = None
    side: Optional[SideMetrics] = None
    scale_mm_per_px: Optional[float] = None
    score: Optional[UBASScore] = None
    overlays: Optional[Dict[str, str]] = None
    pdf_report_b64: Optional[str] = None
    ai_summary: Optional[str] = None
