from fastapi import FastAPI, UploadFile, File
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io

app = FastAPI()

# 🔁 CHANGE THIS PER DEPLOYMENT
MODEL_NAME = "badrex/w2v-bert-2.0-kikuyu-asr"
# MODEL_NAME = "thinkKenya/wav2vec2-large-xls-r-300m-sw"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, 16000
        )

    input_values = processor(
        waveform.squeeze(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return {
        "text": transcription
    }
