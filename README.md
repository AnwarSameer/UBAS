# UBAS Anthropometry (Front + Side) — Complete Build

This is a runnable FastAPI microservice that:
- Accepts **2 mandatory** images (pre-op front, post-op front) and **2 optional** images (pre-op side, post-op side)
- **Auto-detects and crops** the eye region(s) from **any photo** (full-body or close-up) using MediaPipe Face Mesh
- Computes core anthropometric metrics normalized by iris diameter (ID)
- Produces an **objective 0–30 score** (UBAS-FS 30) with subscores
- Generates a PDF one-page report
- Returns a short AI-style summary (local, no external calls by default)

> Note: Some advanced metrics (crease segmentation, lateral hooding area, exact side-view curvature) are approximated or neutral if unavailable. You can replace the stubs with your trained models later.

## Quickstart

### 1) Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 2) Install deps
```bash
pip install -r requirements.txt
```

If OpenCV fails to install on some servers, try `opencv-python-headless` instead of `opencv-python` in `requirements.txt`.

### 3) Run the API
```bash
uvicorn app.main:app --reload
```

- Root: http://127.0.0.1:8000/
- Docs: http://127.0.0.1:8000/docs
- Simple upload UI: http://127.0.0.1:8000/ui

### 4) Example `curl`
```bash
curl -X POST http://127.0.0.1:8000/analyze-multi   -F "pre_front=@/path/pre_front.jpg"   -F "post_front=@/path/post_front.jpg"   -F "pre_side=@/path/pre_side.jpg"   -F "post_side=@/path/post_side.jpg"   -F "use_sticker=false"
```

### 5) VS Code tips
- Open the folder in VS Code
- Select the created virtual environment as Python interpreter
- Create a `launch.json` with a FastAPI/uvicorn configuration or run this in the terminal:
  ```bash
  uvicorn app.main:app --reload
  ```

## Project Layout

```
ubas_complete/
  app/
    __init__.py
    main.py
    schemas.py
    preprocess.py
    inference.py
    qc.py
    metrics.py
    scoring.py
    report.py
    llm_client.py
  static/
    index.html
  requirements.txt
  README.md
```

---

## Extending to Production

- **Replace stubs** in `inference.py` with your ONNX eyelid/brow/crease models.
- **Confidence & CIs**: propagate per-landmark σ from heatmap widths and down-weight low-confidence bins in `scoring.py`.
- **Calibration**: use a 10 mm sticker if available; otherwise, use iris diameter fallback (default 11.8 mm).
- **Security**: disable `/ui` in production, add auth, and set file upload size limits.
- **GPU**: move models to ONNXRuntime GPU or TensorRT as needed.
