import os
import warnings

# Suppress symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

from faster_whisper import WhisperModel

# 22 Official Scheduled Indian Languages + English
INDIC_LANG_MAP = {
    "hi": "Hindi", "mr": "Marathi", "gu": "Gujarati", "bn": "Bengali",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "ml": "Malayalam", "ur": "Urdu", "ne": "Nepali", "sa": "Sanskrit",
    "sd": "Sindhi", "as": "Assamese", "en": "English"
}

# Regional context prompt to prevent phonetic spelling errors (e.g., Parth, Ankit)
REGIONAL_PROMPT = (
    "Parth, Ankit, Aadhaar, UPI, Namaste, Shukriya, Dhanyawad, "
    "Hindi, Marathi, Gujarati, Tamil, Telugu, Kannada, Bengali. "
    "Transcribe accurately in native script."
)

class MultilingualIndianASR:
    def __init__(self, model_size="small", device="cpu"):
        """
        'small' or 'medium' provides high accuracy for Indian regional accents.
        """
        print(f"Loading Multilingual ASR ({model_size}) on {device.upper()}...")
        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    def transcribe(self, audio_path: str, lang_code: str = None, translate_to_en: bool = False):
        if not os.path.exists(audio_path):
            return {"error": f"File '{audio_path}' not found."}

        task = "translate" if translate_to_en else "transcribe"

        # Transcribe with VAD filtering and Indian vocabulary biasing
        segments, info = self.model.transcribe(
            audio_path,
            language=lang_code,          # None = Auto Detect
            task=task,
            beam_size=5,
            vad_filter=True,             # Skips dead air & prevents CPU hangs
            initial_prompt=REGIONAL_PROMPT
        )

        detected_lang = info.language.lower()
        language_name = INDIC_LANG_MAP.get(detected_lang, detected_lang.upper())

        print(f"Detected Language: {language_name} ({info.language_probability:.2%})")

        text_chunks = []
        for segment in segments:
            text_chunks.append(segment.text)

        final_text = " ".join(text_chunks).strip()

        return {
            "language_code": detected_lang,
            "language_name": language_name,
            "confidence": f"{info.language_probability:.2%}",
            "text": final_text
        }


if __name__ == "__main__":
    # Initialize engine
    asr = MultilingualIndianASR(model_size="small", device="cpu")
    
    test_audio = "test.mp3"  # Supply your Hindi/Marathi/Gujarati/English audio
    
    if os.path.exists(test_audio):
        print("\n--- 1. Auto-Detect & Native Script Transcription ---")
        res = asr.transcribe(test_audio)
        print(f"Language : {res['language_name']} ({res['language_code']})")
        print(f"Text     : {res['text']}")

        print("\n--- 2. Direct Translation to English ---")
        res_en = asr.transcribe(test_audio, translate_to_en=True)
        print(f"English  : {res_en['text']}")
    else:
        print(f"\n[Ready] Place an audio file named '{test_audio}' in this folder to test.")
