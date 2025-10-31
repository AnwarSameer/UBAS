# Local stub that returns a concise summary without external calls.
# Replace with your own API integration if desired.

def summarize_with_llm(payload: dict) -> str:
    ubas = payload.get("ubas", {})
    total = ubas.get("total", 0)
    band = ubas.get("band", "Unknown")
    front = payload.get("post", {}).get("front") or payload.get("pre", {}).get("front") or {}
    mrd1 = (front.get("mrd1_L",0)+front.get("mrd1_R",0))/2 if front else 0
    tps = (front.get("tps_mid_L",0)+front.get("tps_mid_R",0))/2 if front else 0
    pfh = (front.get("pfh_L",0)+front.get("pfh_R",0))/2 if front else 0
    return (
        f"Surgery summary: Functional lift (MRD1 avg ~ {mrd1:.2f} ID) with tarsal show ~ {tps:.2f} ID "
        f"and PFH ~ {pfh:.2f} ID. Crease appears near-symmetric; brow position stable. "
        f"Overall rating: {band} (Total {total}/30)."
    )
