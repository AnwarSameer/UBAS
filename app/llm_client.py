import os, json, requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # set this in Render → Environment
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def summarize_with_llm(payload: dict) -> str:
    if not GROQ_API_KEY:
        # safe fallback if not configured
        ubas = payload.get("ubas", {})
        return f"(Local) Overall rating: {ubas.get('band','Unknown')} ({ubas.get('total',0)}/30)."

    messages = [
        {"role": "system",
         "content": ("You are a clinical assistant. Summarize surgical outcomes in 3–5 sentences. "
                     "Highlight MRD1 change, tarsal show (mid/med/lat), crease symmetry, "
                     "brow stability, and side-view sulcus changes. End with the rating band.")},
        {"role": "user",
         "content": "Metrics JSON:\n" + json.dumps(payload, ensure_ascii=False)}
    ]

    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.2},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        # OpenAI-compatible: first choice message content
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(Summary unavailable: {e})"
