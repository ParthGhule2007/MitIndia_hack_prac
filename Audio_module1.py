import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

from faster_whisper import WhisperModel

# Load model
stt_model = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        return f"Error: File '{audio_path}' not found."

    # Define custom names, keywords, and context
   # custom_prompt = "Hello, my name is Parth. Ankit and I are working on this project."

    segments, info = stt_model.transcribe(
        audio_path,
        beam_size=5,            # Keep 3 to 5 for better vocabulary accuracy
        vad_filter=True,
       #  # Forces model to bias towards these words
        language="en"           # Enforce language if speaking Indian English
    )

    print(f"Detected language: '{info.language}' ({info.language_probability:.2%})")

    text_chunks = []
    for segment in segments:
        text_chunks.append(segment.text)

    return " ".join(text_chunks).strip()

if __name__ == "__main__":
    audio_file = "test.mp3"
    if os.path.exists(audio_file):
        print("Transcribing with custom vocabulary...")
        print(transcribe_audio(audio_file))
