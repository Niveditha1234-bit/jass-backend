# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import numpy as np
import os
import time
from typing import List
from dotenv import load_dotenv
try:
    import faiss
except ImportError:
    faiss = None

# Load environment variables
load_dotenv()

app = FastAPI()

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- GEMINI SETUP --------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=GEMINI_API_KEY)

# Rate limiting (basic protection)
last_request_time = 0
MIN_REQUEST_INTERVAL = 2  # seconds

# -------------------- KNOWLEDGE BASE --------------------
KNOWLEDGE_BASE = [
    "Niveditha P is an entry-level AI/ML Engineer from Bengaluru, India. Contact: niveditha.prakash21@gmail.com",
    "Niveditha's LinkedIn: linkedin.com/in/niveditha-p-5130a02b1, GitHub: github.com/Niveditha1234-bit",
    "She is pursuing B.E. in Artificial Intelligence and Data Science at KSSEM, graduating in 2026 with CGPA 8.48",
    "Her skills include Python, SQL, Machine Learning, Deep Learning, NLP, Computer Vision, FastAPI, and TensorFlow",

    "Deepfake Detection System: Built using CNN and Xception, trained on 150k+ images and 7k videos, achieving 89% accuracy",
    "Smart Attendance System: Uses QR code validation and LBPH-based face recognition with Flask backend",
    "Indian Sign Language Translator: Built using TensorFlow and MediaPipe with real-time gesture recognition",

    "She is a co-author of an IEEE ICCCNP 2025 conference paper on Deepfake Detection",
    "She led a 24-hour hackathon with 150+ participants as Hackathon Coordinator at KSSEM"
]

# -------------------- FAISS SETUP --------------------
def simple_embedding(text: str) -> np.ndarray:
    hash_val = hash(text.lower())
    np.random.seed(abs(hash_val) % (2**32))
    return np.random.randn(384).astype("float32")

dimension = 384
index = faiss.IndexFlatL2(dimension)
embeddings = np.array([simple_embedding(doc) for doc in KNOWLEDGE_BASE])
index.add(embeddings)

def retrieve_context(query: str, k: int = 5) -> List[str]:
    query_embedding = simple_embedding(query).reshape(1, -1)
    _, indices = index.search(query_embedding, k)
    return [KNOWLEDGE_BASE[i] for i in indices[0]]

# -------------------- API MODELS --------------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# -------------------- CHAT ENDPOINT --------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global last_request_time

    try:
        # Rate limiting
        now = time.time()
        if now - last_request_time < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - (now - last_request_time))

        context_docs = retrieve_context(request.message)
        context = "\n".join(context_docs)

        prompt = f"""
You are Jass, an AI assistant for Niveditha's portfolio.

Context:
{context}

User question:
{request.message}

Rules:
- Answer only from context
- Be friendly and concise (2–3 sentences)
- If info is missing, say you don't know

Answer:
"""

        # ✅ ONLY USE VERIFIED WORKING MODELS
        models_to_try = [
            "gemini-flash-latest",
            "gemini-pro-latest",
            "gemini-2.5-flash"
        ]

        response = None
        last_error = None

        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                break
            except Exception as e:
                last_error = e

        if response is None:
            raise last_error

        last_request_time = time.time()
        return ChatResponse(response=response.text.strip())

    except Exception as e:
        msg = str(e)
        if "quota" in msg.lower() or "429" in msg:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
        raise HTTPException(status_code=500, detail=msg)

# -------------------- HEALTH --------------------
@app.get("/")
async def root():
    return {"message": "Jass Backend API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/models")
async def list_models():
    models = client.models.list()
    return {"models": [m.name for m in models]}

# -------------------- RUN --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
