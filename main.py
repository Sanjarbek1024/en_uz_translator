import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.model import (
    Attention,
    Decoder,
    Encoder,
    Seq2Seq,
    translate_sentence,
)

# Paths
# BASE_DIR = folder where main.py lives (the project root)
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "saved_model" / "full_seq2seq_checkpoint.pth"
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"
STATIC_DIR = BASE_DIR / "static"

# Create static/ if it doesn't exist yet
STATIC_DIR.mkdir(exist_ok=True)

# Global state
state: dict = {}


# Lifespan: load model once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if not CHECKPOINT_PATH.exists():
        print(f"[WARNING] Checkpoint not found at {CHECKPOINT_PATH}")
        print("[INFO] API will start but /translate will return an error until model is placed.")
        state["model"] = None
        state["device"] = device
        state["input_word2index"] = {}
        state["output_index2word"] = {}
        state["max_length"] = 30
    else:
        print(f"[INFO] Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        embed_size = checkpoint["embed_size"]
        hidden_size = checkpoint["hidden_size"]
        num_layers = checkpoint["num_layers"]
        dropout = checkpoint["dropout"]
        input_vocab_size = checkpoint["input_vocab_size"]
        output_vocab_size = checkpoint["output_vocab_size"]
        max_length = checkpoint.get("max_length", 30)

        attention = Attention(hidden_size)

        encoder = Encoder(
            input_dim=input_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        decoder = Decoder(
            output_dim=output_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            attention=attention,
        )

        model = Seq2Seq(encoder, decoder, device).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        state["model"] = model
        state["device"] = device
        state["input_word2index"] = checkpoint["input_lang_word2index"]
        state["output_index2word"] = checkpoint["output_lang_index2word"]
        state["max_length"] = max_length

        print("[INFO] Model loaded successfully ✓")

    yield

    state.clear()


# App
app = FastAPI(
    title="EN→UZ Translator",
    description="Seq2Seq GRU + Attention translation model",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Schemas
class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    input: str
    translation: str
    device: str


# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if state.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Place full_seq2seq_checkpoint.pth in saved_model/ and restart.",
        )

    translation = translate_sentence(
        model=state["model"],
        sentence=req.text,
        input_word2index=state["input_word2index"],
        output_index2word=state["output_index2word"],
        device=state["device"],
        max_length=state["max_length"],
    )

    return TranslateResponse(
        input=req.text,
        translation=translation,
        device=str(state["device"]),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.get("model") is not None,
        "device": str(state.get("device", "unknown")),
    }

