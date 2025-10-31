import os, json, requests

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")

def summarize_with_llm(payload: dict) -> str:
    if not LLM_ENDPOINT or not LLM_API_KEY:
        # Safe fallback if env vars aren't set
        ubas = payload.get("ubas", {})
        total = ubas.get("total", 0)
        band = ubas.get("band", "Unknown")
        return f"(Local) Overall rating: {band} ({total}/30)."
    try:
        r = requests.post(
            LLM_ENDPOINT,
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "instruction": (
                  "Summarize the surgical outcome in 3–5 sentences. "
                  "Highlight MRD1 change, tarsal show (mid/med/lat), crease symmetry, "
                  "brow stability, side-view sulcus changes. End with the rating band."
                ),
                "metrics": payload
            }),
            timeout=20
        )
        r.raise_for_status()
        return r.json().get("summary", "(No summary field in response)")
    except Exception as e:
        return f"(Summary unavailable: {e})"
