import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from groq import Groq

# 1. Setup Device (GPU/CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 ML Models initialized on device: {DEVICE}")

class Transcriber:
    """
    Uses Groq API to transcribe audio.
    Requires API key to be passed at runtime.
    """
    def __init__(self, api_key: str, model_name="whisper-large-v3"):
        self.api_key = api_key
        if not self.api_key:
            print("⚠️ WARNING: No API Key provided. Transcription will fail.")
            self.client = None
        else:
            # Initialize Groq client
            self.client = Groq(api_key=self.api_key)
        
        self.model = model_name

    def transcribe(self, audio_path: str) -> list:
        """
        Transcribes audio file using Groq.
        Returns a list of segments: [{'text': '...', 'start': 0.0, 'end': 1.0}, ...]
        """
        if not self.client:
            print("❌ Groq client not initialized. Cannot transcribe.")
            return []

        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return []

        try:
            print(f"🎙️ Transcribing {os.path.basename(audio_path)} with Groq...")
            
            with open(audio_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), file.read()),
                    model=self.model,
                    temperature=0.0,
                    response_format="verbose_json", # Essential for timestamps
                )
            
            # Groq returns an object with a .segments list. 
            # We convert it to a clean dictionary format for our pipeline.
            segments = []
            if hasattr(transcription, 'segments'):
                for seg in transcription.segments:
                    segments.append({
                        "text": seg['text'].strip(),
                        "start": seg['start'],
                        "end": seg['end']
                    })
            return segments

        except Exception as e:
            print(f"❌ Groq Transcription Error: {e}")
            return []


class Captioner:
    """
    Uses local Salesforce BLIP model to caption images.
    """
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        print(f"🖼️ Loading BLIP model: {model_name}...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(DEVICE)

    def caption_image(self, image_path: str) -> str:
        """
        Generates a caption for a single image file.
        """
        if not os.path.exists(image_path):
            print(f"❌ Image missing: {image_path}")
            return ""

        try:
            raw_image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs
            inputs = self.processor(raw_image, return_tensors="pt").to(DEVICE)
            
            # Generate caption
            out = self.model.generate(**inputs, max_new_tokens=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            print(f"❌ Captioning Error ({os.path.basename(image_path)}): {e}")
            return ""


class Embedder:
    """
    Uses local Sentence Transformers to create vector embeddings.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"🧠 Loading Embedder: {model_name}...")
        self.model = SentenceTransformer(model_name, device=DEVICE)

    def embed_batch(self, texts: list) -> list:
        """
        Embeds a list of strings. Returns a list of list-of-floats.
        """
        if not texts:
            return []
            
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            # Convert numpy array to list (required for ChromaDB/Qdrant)
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
            return embeddings
            
        except Exception as e:
            print(f"❌ Embedding Error: {e}")
            return []