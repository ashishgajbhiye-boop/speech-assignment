import whisper

def transcribe_whisper(audio_path: str) -> str:
    model = whisper.load_model("base")  # small, fast
    result = model.transcribe(audio_path)
    return result["text"]