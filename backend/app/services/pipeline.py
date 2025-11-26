import os
import shutil
from pathlib import Path
# Use relative imports within the 'services' package
from .video_processor import VideoProcessor
from .model import Transcriber, Captioner, Embedder
from .qdrant_db import QdrantDB
import uuid
import yt_dlp

class YouTubePipeline:

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key

        base_dir = Path(__file__).resolve().parent.parent.parent  # backend/
        self.data_dir = base_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.keyframe_dir = self.data_dir / "keyframes"
        self.keyframe_dir.mkdir(exist_ok=True)
        
        self.video_dir = self.data_dir / "video"
        self.video_dir.mkdir(exist_ok=True)
        
        self.audio_dir = self.data_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)


        # Final output files (standardized names)
        # We'll get the video ID to make names unique
        self.video_id = self.get_video_id(url)
        self.video_file = self.video_dir / f"{self.video_id}.mp4"
        self.audio_file = self.audio_dir / f"{self.video_id}.webm"

        # Components
        # Pass the specific output dirs to the processor
        self.vp = VideoProcessor(
            url=url,
            output_dir_keyframes=self.keyframe_dir,
            output_dir_audio=self.audio_dir,
            output_dir_video=self.video_dir
        )
        self.transcriber = Transcriber(api_key=self.api_key)
        self.captioner = Captioner()
        self.embedder = Embedder()
        self.db = QdrantDB()

    def get_video_id(self, url):
        """Helper to get video ID for unique file naming."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('id', 'default_video')
        except Exception:
            return f"video_{uuid.uuid4().hex[:8]}"


    # ---------------------------------------------------------
    # FORCE COPY/MOVE FILE TO STANDARDIZED PATH
    # ---------------------------------------------------------
    def _standardize_path(self, src_path: str, target_path: Path) -> str:
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file {src} not found.")
            
        if target_path.exists():
            target_path.unlink()

        # Move the file
        shutil.move(src, target_path)
        return str(target_path)

    # ---------------------------------------------------------
    # MAIN PIPELINE
    # ---------------------------------------------------------
    def run(self):

        print("\n🚀 Starting YouTube Pipeline\n")

        # -----------------------------
        # STEP 1 — Download Video
        # -----------------------------
        raw_video_path, title, duration = self.vp.download_video(self.url)
        # Standardize the name (e.g., move 123.mkv to {video_id}.mp4)
    def get_video_id(self, url):
        """Helper to get video ID for unique file naming."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('id', 'default_video')
        except Exception:
            return f"video_{uuid.uuid4().hex[:8]}"


    # ---------------------------------------------------------
    # FORCE COPY/MOVE FILE TO STANDARDIZED PATH
    # ---------------------------------------------------------
    def _standardize_path(self, src_path: str, target_path: Path) -> str:
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file {src} not found.")
            
        # If source and target are the same, we are done
        if src.resolve() == target_path.resolve():
            return str(target_path)

        if target_path.exists():
            target_path.unlink()

        # Move the file
        shutil.move(src, target_path)
        return str(target_path)

    # ---------------------------------------------------------
    # MAIN PIPELINE
    # ---------------------------------------------------------
    def run(self):

        print("\n🚀 Starting YouTube Pipeline\n")

        # -----------------------------
        # STEP 1 — Download Video
        # -----------------------------
        raw_video_path, title, duration = self.vp.download_video(self.url)
        # Standardize the name (e.g., move 123.mkv to {video_id}.mp4)
        # Note: download_video already tries to get .mp4, but this ensures it
        final_video_path = self._standardize_path(raw_video_path, self.video_file)
        print(f"🎬 Saved final video at {final_video_path}")

        # -----------------------------
        # STEP 2 — Download Audio
        # -----------------------------
        # This vp method is already set to download as webm
        final_audio_path = self.vp.download_audio_webm(self.url, out_dir=self.audio_dir)
        # Standardize path in case download_audio_webm implementation changes
        final_audio_path = self._standardize_path(final_audio_path, self.audio_file)
        print(f"🎧 Saved final audio at {final_audio_path}")

        # -----------------------------
        # STEP 3 — Extract Keyframes
        # -----------------------------
        # We pass the path to the *final* video file
        keyframes = self.vp.process(
            url=self.url,
            scene_threshold=25.0,
            min_interval=2,
            max_interval=15
        )
        print(f"🖼️ Extracted {len(keyframes)} keyframes")

        # -----------------------------
        # STEP 4 — Transcription
        # -----------------------------
        print("🎙️ Transcribing audio...")
        segments = self.transcriber.transcribe(final_audio_path)
        print(f"📄 {len(segments)} transcript segments")

        # -----------------------------
        # STEP 5 — Embed Transcript
        # -----------------------------
        texts = [s["text"] for s in segments]
        text_emb = self.embedder.embed_batch(texts)
        transcript_metadata = [
            {
                "type": "transcript",
                "timestamp": int(s["start"]),
            }
            for s in segments
        ]
        
        if text_emb:
            self.db.add_embeddings(
                embeddings=text_emb,
                documents=texts,
                metadatas=transcript_metadata
            )


        # -----------------------------
        # STEP 6 — Caption Keyframes
        # -----------------------------
        print("🖼️ Captioning keyframes...")
        cap_texts = []
        cap_metas = []

        for kf in keyframes:
            caption = self.captioner.caption_image(kf["filepath"])
            if not caption:
                continue

            cap_texts.append(caption)
            cap_metas.append({
                "type": "frame",
                "timestamp": int(kf["timestamp"]),
                "filename": kf["filename"]
            })


        print(f"📝 Generated {len(cap_texts)} captions")

        # -----------------------------
        # STEP 7 — Embed Captions
        # -----------------------------
        cap_emb = self.embedder.embed_batch(cap_texts)

        if cap_emb:
            self.db.add_embeddings(
                embeddings=cap_emb,
                documents=cap_texts,
                metadatas=cap_metas
            )

        print("\n🎉 Pipeline Completed Successfully!\n")

        return {
            "video_file": str(self.video_file),
            "audio_file": str(self.audio_file),
            "keyframes": len(keyframes),
            "segments": len(segments)
        }