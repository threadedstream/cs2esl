from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import soundfile as sf
import torch
import numpy as np

torch.serialization.add_safe_globals([
    np.core.multiarray.scalar,
    np.ndarray,
    np.dtype,
])

from bark import generate_audio, preload_models

app = FastAPI()

preload_models()

class TTSRequest(BaseModel):
    text: str
    emotion: str = "hype"  # hype | tense | calm

EMOTION_PROMPTS = {
    "hype": "Excited esports commentator voice. High energy. Crowd roaring.",
    "tense": "Low, tense esports caster voice. Controlled breathing.",
    "calm": "Calm analyst voice. Confident and composed."
}

@app.post("/tts")
def tts(req: TTSRequest):
    prompt = f"{EMOTION_PROMPTS.get(req.emotion, '')} {req.text}"

    audio = generate_audio(prompt)

    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format="WAV")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")
