from fastapi import FastAPI
from pydantic import BaseModel
import torch
import soundfile as sf
import io
import base64

from transformers import AutoProcessor, AutoModel

app = FastAPI()

MODEL_NAME = "kyutai/pocket-tts"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
def tts(req: TTSRequest):
    with torch.no_grad():
        inputs = processor(req.text, return_tensors="pt")
        audio = model.generate(**inputs)

    # Convert tensor → wav bytes
    wav_io = io.BytesIO()
    sf.write(wav_io, audio.cpu().numpy(), samplerate=24000, format="WAV")
    wav_bytes = wav_io.getvalue()

    return {
        "audio_base64": base64.b64encode(wav_bytes).decode("utf-8")
    }

@app.get("/health")
def health():
    return {"status": "ok"}