from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional
import base64, cv2

from .schemas import AnalyzeResponse, Calibration
from .preprocess import preprocess_any
from .inference import run_front_pipeline, run_side_pipeline
from .qc import run_qc
from .metrics import front_metrics, side_metrics
from .scoring import score
from .report import make_pdf
from .llm_client import summarize_with_llm

app = FastAPI(title="UBAS Anthropometry", version="1.0.0")

@app.get("/", response_class=JSONResponse)
def root():
    return {"ok": True, "name": "UBAS Anthropometry", "docs": "/docs", "ui": "/ui"}

@app.get("/ui", response_class=HTMLResponse)
def ui():
    html = open(__file__.replace("main.py", " ../static/index.html").replace(" ", "")).read()
    return HTMLResponse(html)

def _to_b64(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")

@app.post("/analyze-multi")
async def analyze_multi(
    pre_front: UploadFile = File(..., description="Pre-op both eyes, front"),
    post_front: UploadFile = File(..., description="Post-op both eyes, front"),
    pre_side: Optional[UploadFile] = File(None, description="Pre-op side (optional)"),
    post_side: Optional[UploadFile] = File(None, description="Post-op side (optional)"),
    use_sticker: bool = Form(False),
    sticker_px: Optional[float] = Form(None),
    sticker_mm: float = Form(10.0),
    iris_diam_mm: float = Form(11.8)
):
    # 1) Auto-crop to standardized frames
    pf_img, pf_meta = preprocess_any(await pre_front.read(), view="front")
    qf_img, qf_meta = preprocess_any(await post_front.read(), view="front")
    ps_img, ps_meta = (None, None)
    qs_img, qs_meta = (None, None)
    if pre_side is not None:
        ps_img, ps_meta = preprocess_any(await pre_side.read(), view="side")
    if post_side is not None:
        qs_img, qs_meta = preprocess_any(await post_side.read(), view="side")

    # 2) Landmarking/segmentation on cropped images
    pf_b64 = _to_b64(pf_img); qf_b64 = _to_b64(qf_img)
    f_img0, f_lm_pre, roll_pre = run_front_pipeline(pf_b64)
    f_img1, f_lm_post, roll_post = run_front_pipeline(qf_b64)

    s_metrics_pre = None; s_metrics_post = None
    if ps_img is not None:
        s_img0, s_feat_pre = run_side_pipeline(_to_b64(ps_img))
        s_metrics_pre = side_metrics(s_feat_pre)
    if qs_img is not None:
        s_img1, s_feat_post = run_side_pipeline(_to_b64(qs_img))
        s_metrics_post = side_metrics(s_feat_post)

    # 3) QC (use post front for accept)
    qc = run_qc(cv2.cvtColor(f_img1, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(f_img1, cv2.COLOR_BGR2GRAY),
                f_lm_post, roll_post)
    if not qc.passed:
        return {"qc": qc.dict(), "message": "Retake required", "overlays": None}

    # 4) Calibration config
    calib_mode = "sticker" if use_sticker and sticker_px else "iris"
    calib = Calibration(mode=calib_mode, iris_diam_mm=iris_diam_mm,
                        sticker_diam_mm=sticker_mm, sticker_px=sticker_px)

    # 5) Compute metrics (demo uses same landmarks for L/R; replace with per-eye if desired)
    front_pre, mm_px_pre = front_metrics(f_lm_pre, f_lm_pre, calib)
    front_post, mm_px_post = front_metrics(f_lm_post, f_lm_post, calib)
    side_post = s_metrics_post if s_metrics_post else None

    # 6) Score (post-op focus)
    ubas = score(front_post, side_post, preop=None)

    # 7) AI summary (local stub)
    llm_payload = {
        "pre": {"front": front_pre.dict(), "side": s_metrics_pre.dict() if s_metrics_pre else None},
        "post": {"front": front_post.dict(), "side": s_metrics_post.dict() if s_metrics_post else None},
        "ubas": ubas.dict()
    }
    ai_summary = summarize_with_llm(llm_payload)

    # 8) PDF
    pdf_b64 = make_pdf({
        "Total": ubas.total, "Band": ubas.band,
        "TPS mid (post avg ID)": round((front_post.tps_mid_L+front_post.tps_mid_R)/2, 3),
        "MRD1 (post avg ID)": round((front_post.mrd1_L+front_post.mrd1_R)/2, 3),
        "PFH (post avg ID)": round((front_post.pfh_L+front_post.pfh_R)/2, 3),
    })

    return {
        "qc": qc.dict(),
        "ubas": ubas.dict(),
        "ai_summary": ai_summary,
        "scale_mm_per_px_post": mm_px_post,
        "debug_overlays": {
            "pre_front_crop_png_b64": pf_b64,
            "post_front_crop_png_b64": qf_b64
        },
        "pdf_report_b64": pdf_b64
    }
