from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os
import json
import numpy as np
import requests
import faiss
from PyPDF2 import PdfReader
from functools import lru_cache
import uvicorn

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class LimitRequestSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int = 20_000_000):  # ~20 MB
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_upload_size:
            return Response("Request too large", status_code=413)
        return await call_next(request)

# --- Setup FastAPI App ---
app = FastAPI()  # ✅ restored with Swagger support

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directories ---
UPLOAD_DIR = "pdfs"
TEXT_DIR = "data"
INDEX_DIR = "index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# --- Lazy model loader ---
@lru_cache()
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- Schema ---
class AskRequest(BaseModel):
    question: str
    doc_id: str = ""

# --- PDF Parsing ---
def parse_text(file_path):
    chunks = []
    reader = PdfReader(file_path)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            cleaned = text.strip().replace("\n", " ")
            chunks.append(cleaned)
    return chunks

def embed_and_index(doc_id, chunks):
    model = get_model()
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(INDEX_DIR, f"{doc_id}.index"))
    with open(os.path.join(TEXT_DIR, f"{doc_id}.json"), "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f)

# --- LLM Call ---
def call_mistral(prompt):
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })
    return res.json().get("response", "[No response]")

# --- Endpoints ---
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile):
    ext = os.path.splitext(file.filename)[-1]
    doc_id = os.path.splitext(file.filename)[0].replace(" ", "_")
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    chunks = parse_text(file_path)
    embed_and_index(doc_id, chunks)

    return {"doc_id": doc_id, "text_chunks": len(chunks)}

@app.get("/list_docs")
def list_docs():
    return {
        "docs": [f.replace(".json", "") for f in os.listdir(TEXT_DIR) if f.endswith(".json")]
    }

@app.post("/ask")
async def ask_doc(data: AskRequest):
    model = get_model()
    query_embed = model.encode([data.question])
    doc_id = data.doc_id

    index_path = os.path.join(INDEX_DIR, f"{doc_id}.index")
    json_path = os.path.join(TEXT_DIR, f"{doc_id}.json")

    if not os.path.exists(index_path) or not os.path.exists(json_path):
        return JSONResponse(status_code=404, content={"error": "Document not indexed."})

    index = faiss.read_index(index_path)
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    D, I = index.search(np.array(query_embed), k=1)
    context = chunks[I[0][0]]

    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question: {data.question}
Answer:"""
    answer = call_mistral(prompt)

    return {
        "answer": answer,
        "doc_id": doc_id,
        "context": context,
        "pdf_url": f"/static/{doc_id}.pdf"
    }

@app.get("/pdfs/{filename}")
def serve_pdf(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "PDF not found"})
    return FileResponse(path, media_type="application/pdf")

# --- Run App ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
