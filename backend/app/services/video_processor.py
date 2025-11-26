import cv2
import os
from scenedetect import detect, ContentDetector
import yt_dlp
import matplotlib.pyplot as plt
from PIL import Image
from datetime import timedelta
import subprocess
from pathlib import Path
import datetime


# If you use cookies, set COOKIE_PATH to a Path() here. If not, leave as Path("").
COOKIE_PATH = ""  # adjust if you keep cookies somewhere else


class VideoProcessor:
    def __init__(self,
                 url: str = None,
                 output_dir_keyframes: Path | str = None,
                 output_dir_audio: Path | str = None,
                 output_dir_video: Path | str = None):
        self.url = url

        # Compute canonical project/data directories relative to this file (services/)
        services_dir = Path(__file__).resolve().parent
        project_root = services_dir.parent  # backend/
        data_dir = project_root / "data"

        # Defaults (can be overridden by passing args)
        self.output_dir_keyframes = Path(output_dir_keyframes) if output_dir_keyframes else (data_dir / "keyframes")
        self.output_dir_audio = Path(output_dir_audio) if output_dir_audio else (data_dir / "audio")
        self.output_dir_video = Path(output_dir_video) if output_dir_video else (data_dir / "video")

        # Ensure dirs exist
        self.output_dir_keyframes.mkdir(parents=True, exist_ok=True)
        self.output_dir_audio.mkdir(parents=True, exist_ok=True)
        self.output_dir_video.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. DOWNLOAD VIDEO
    # ------------------------------------------------------------------
    def download_video(self, url: str) -> tuple:
        """Download YouTube video using Android client. Returns (video_file_path, title, duration)."""

        # Build an output template inside the video dir so yt-dlp writes a file (not just a folder)
        outtmpl = str(self.output_dir_video / "%(id)s.%(ext)s")

        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': outtmpl,
            'quiet': True,
            'no_warnings': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android'],
                }
            },
        }

        print(f"📥 Downloading: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            # yt-dlp will fill %(id)s and extension; extract resulting filename if possible
            # The downloader saved to outtmpl with metadata; we can build expected filename:
            video_id = info.get('id')
            ext = info.get('ext') or "mp4"
            downloaded_path = self.output_dir_video / f"{video_id}.{ext}"

        print(f"✅ Downloaded: {title} ({duration}s) -> {downloaded_path}")
        return str(downloaded_path), title, duration

    # ------------------------------------------------------------------
    # 2. DETECT SCENES
    # ------------------------------------------------------------------
    def detect_scenes(self, video_path: str, threshold=27.0):
        print(f"🎬 Detecting scenes (threshold={threshold})...")
        scene_list = detect(video_path, ContentDetector(threshold=threshold))

        scenes = []
        for i, scene in enumerate(scene_list):
            start = scene[0].get_seconds()
            end = scene[1].get_seconds()
            scenes.append({
                'id': i,
                'start': start,
                'end': end,
                'duration': end - start
            })

        print(f"✅ Found {len(scenes)} scenes")
        return scenes

    # ------------------------------------------------------------------
    # 3. EXTRACT KEYFRAMES
    # ------------------------------------------------------------------
    def extract_keyframes(self, video_path: str, scene_threshold=27.0,
                           min_interval=20, max_interval=60):
        # ensure keyframes dir exists (already created in __init__ but safe)
        self.output_dir_keyframes.mkdir(parents=True, exist_ok=True)

        # Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps and fps > 0 else 0
        cap.release()

        print(f"📹 Video: {duration:.1f}s, {fps:.1f} fps")

        # Detect scenes
        scenes = self.detect_scenes(video_path, threshold=scene_threshold)
        scene_times = [scene['start'] for scene in scenes]

        # Time-based keyframes
        time_based = list(range(0, int(duration) + 1, max_interval)) if duration > 0 else [0]

        # Combine + filter
        all_times = sorted(set(scene_times + time_based))
        filtered = [all_times[0]] if all_times else [0]

        for t in all_times[1:]:
            if t - filtered[-1] >= min_interval:
                filtered.append(t)

        print(f"🎯 Extracting {len(filtered)} keyframes...")

        keyframes = []
        cap = cv2.VideoCapture(video_path)

        for idx, timestamp in enumerate(filtered):
            frame_num = int(timestamp * fps) if fps and fps > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if ret:
                filename = f"frame_{idx:03d}_{timestamp:.1f}s.jpg"
                filepath = self.output_dir_keyframes / filename
                cv2.imwrite(str(filepath), frame)

                keyframes.append({
                    'index': idx,
                    'timestamp': timestamp,
                    'timestamp_formatted': str(timedelta(seconds=int(timestamp))),
                    'filepath': str(filepath),
                    'filename': filename
                })

        cap.release()
        print(f"✅ Saved {len(keyframes)} keyframes to '{self.output_dir_keyframes}/'")

        return keyframes, scenes

    # ------------------------------------------------------------------
    # 4. VISUALIZE KEYFRAMES
    # ------------------------------------------------------------------
    def visualize_keyframes(self, keyframes, max_display=12):
        num_frames = min(len(keyframes), max_display)
        cols = 4
        rows = (num_frames + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for idx in range(rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            if idx < num_frames:
                kf = keyframes[idx]
                img = Image.open(kf['filepath'])
                ax.imshow(img)
                ax.set_title(f"{kf['index']}: {kf['timestamp_formatted']}", fontsize=10, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 5. PRINT SUMMARY
    # ------------------------------------------------------------------
    def print_summary(self, keyframes, scenes):
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)

        print(f"\n🎬 Scenes: {len(scenes)}")
        for scene in scenes[:5]:
            print(f"  Scene {scene['id']}: {scene['start']:.1f}s - {scene['end']:.1f}s")
        if len(scenes) > 5:
            print(f"  ... and {len(scenes) - 5} more")

        print(f"\n🎯 Keyframes: {len(keyframes)}")
        if keyframes:
            first = keyframes[0]['timestamp']
            last = keyframes[-1]['timestamp']
            avg_gap = (last - first) / (len(keyframes) - 1) if len(keyframes) > 1 else 0
            print(f"  First: {keyframes[0]['timestamp_formatted']}")
            print(f"  Last: {keyframes[-1]['timestamp_formatted']}")
            print(f"  Avg gap: {avg_gap:.1f}s")

        print(f"\n📁 Files:")
        for kf in keyframes[:5]:
            print(f"  {kf['filename']}")
        if len(keyframes) > 5:
            print(f"  ... and {len(keyframes) - 5} more")

        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # 6. MAIN PIPELINE
    # ------------------------------------------------------------------
    def process(self, url=None, scene_threshold=27.0, min_interval=20, max_interval=60):
        print("🚀 Starting pipeline...\n")

        url = url or self.url
        if not url:
            raise ValueError("No YouTube URL provided.")

        video_path, title, duration = self.download_video(url)

        keyframes, scenes = self.extract_keyframes(
            video_path,
            scene_threshold=scene_threshold,
            min_interval=min_interval,
            max_interval=max_interval
        )

        self.print_summary(keyframes, scenes)
        # self.visualize_keyframes(keyframes)

        print(keyframes)

        result = [
            {
                "index": item["index"],
                "timestamp": int(item["timestamp"]),
                "filename": item["filename"],
                "filepath": item["filepath"]
            }
            for item in keyframes
        ]

        print(result)

        return result
    
    def download_audio_webm(self, url: str, out_dir: Path | str | None = None) -> str:
        """
        Download audio and convert to WebM format if needed.
        Returns the absolute path to the saved WEBM file as a string.
        """
        out_dir = Path(out_dir) if out_dir else Path(self.output_dir_audio)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Download with original extension first
        outtmpl = str(out_dir / "%(id)s.%(ext)s")
        
        ydl_opts = {
            # Just get the best audio available
            'format': 'bestaudio/best',
            'outtmpl': outtmpl,
            'quiet': False,  # Set to False to see what's happening
            'no_warnings': False,
            
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],
                }
            },
        }
        
        print(f"🎧 Downloading audio: {url}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_id = info.get('id')
            ext = info.get('ext', 'unknown')
            
            # Find the downloaded file
            downloaded_file = None
            for file in out_dir.glob(f"{audio_id}.*"):
                downloaded_file = file
                break
            
            if not downloaded_file:
                raise FileNotFoundError(f"Downloaded file not found for ID: {audio_id}")
            
            print(f"✅ Downloaded: {downloaded_file}")
            
            # Convert to WebM if not already
            if downloaded_file.suffix.lower() != '.webm':
                print(f"🔄 Converting {downloaded_file.suffix} to WebM...")
                webm_path = out_dir / f"{audio_id}.webm"
                
                import subprocess
                result = subprocess.run([
                    'ffmpeg', '-i', str(downloaded_file),
                    '-c:a', 'libopus',
                    '-b:a', '96k',
                    '-vn',  # No video
                    str(webm_path),
                    '-y'
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"⚠️ FFmpeg conversion failed: {result.stderr}")
                    # Return original file if conversion fails
                    return str(downloaded_file)
                
                # Delete original file
                downloaded_file.unlink()
                print(f"✅ Converted to WebM: {webm_path}")
                return str(webm_path)
            
            return str(downloaded_file)
    
    
    def quick_test(self, test_url=None):
        print("=" * 70)
        print("🧪 QUICK TEST - VideoProcessor")
        print("=" * 70 + "\n")

        url = test_url or self.url
        if not url:
            raise ValueError("No test URL provided.")

        keyframe_dict = self.process(
            url,
            scene_threshold=25.0,
            min_interval=2,
            max_interval=15
        )

        # optionally download audio to data/audio
        self.download_audio_webm(url)
