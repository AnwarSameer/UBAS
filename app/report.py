import base64, io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def make_pdf(summary: dict) -> str:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 800, "UBAS-FS 30 Report")
    c.setFont("Helvetica", 11)
    y = 770
    for k, v in summary.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 16
    c.showPage(); c.save()
    return base64.b64encode(buf.getvalue()).decode("utf-8")
