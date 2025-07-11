import os, json, time, numpy as np
import gradio as gr
try:
    import openai
except ModuleNotFoundError:
    raise RuntimeError("Install openai: pip install openai")
openai.api_key = os.getenv("OPENAI_API_KEY")

# —————————————————————————————————————————
# 1) Load & embed once on startup
with open("candidates.json","r",encoding="utf-8") as f:
    records = json.load(f)

def embed(text):
    return np.array(
        openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text[:8000]
        )["data"][0]["embedding"],
        dtype=np.float32
    )

people = []
for row in records:
    d = row.get("data",{})
    raw = (d.get("rawText") or d.get("summary") or "").strip()
    if not raw: continue
    # metadata
    name = (d.get("candidateName") or ["Unknown"])[0]
    email= (d.get("email") or [""])[0]
    loc  = ", ".join(v for v in (d.get("location") or {}).values() if isinstance(v,str))
    skills = ", ".join(
        s if isinstance(s,str) else next((v for v in s.values() if isinstance(v,str)),"")
        for s in (d.get("skill") or [])
    )
    vec = embed(raw)
    people.append({"vec":vec,"name":name,"email":email,"location":loc,"skills":skills})

# —————————————————————————————————————————
# 2) Search function
def search_ui(query, location_filter):
    qv = embed(query)
    sims = []
    for p in people:
        if location_filter and location_filter.lower() not in p["location"].lower():
            continue
        score = float(np.dot(p["vec"],qv) /
                      (np.linalg.norm(p["vec"])*np.linalg.norm(qv)))
        sims.append((score,p))
    sims.sort(key=lambda x: x[0], reverse=True)
    # build table of top 5
    out = []
    for score,p in sims[:5]:
        # preview first 5 skills
        skills = ", ".join(p["skills"].split(",")[:5])
        out.append({
            "Score": f"{score:.3f}",
            "Name":  p["name"],
            "Email": p["email"],
            "Location": p["location"],
            "Skills": skills + ("..." if len(p["skills"].split(","))>5 else "")
        })
    return out

# —————————————————————————————————————————
# 3) Gradio interface
iface = gr.Interface(
    fn=search_ui,
    inputs=[
      gr.Textbox(lines=2, label="Search query"),
      gr.Textbox(lines=1, label="Location filter (optional)")
    ],
    outputs=gr.Dataframe(label="Top 5 candidates"),
    title="AI Recruiter",
    description="Enter a semantic search + optional location to find best-fit candidates."
)

if __name__ == "__main__":
    iface.launch()
