# Jass Backend - Portfolio Chatbot API

Backend API for Niveditha's portfolio chatbot using FastAPI, FAISS, and Gemini AI.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variable:
```bash
export GEMINI_API_KEY="your-api-key"
```

3. Run server:
```bash
python main.py
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/chat` - Chat endpoint

## Deployment

Deploy to Render.com (free tier)