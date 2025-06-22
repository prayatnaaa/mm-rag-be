import os
import cv2
import tempfile
import yt_dlp
from fastapi import HTTPException
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel

from app.utils.file_utils import save_frame
from app.retriever.embed_clip import embed_text_image
from app.retriever.faiss_index import add_embedding, save_faiss_indices
from app.db.metadata_store import save_source
from app.utils.minio_client import upload_image


def transcribe_audio_whisper(audio_path: str):
    print(f"[INFO] Transcribing audio: {audio_path}")
    model = WhisperModel("base", compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    return [{"text": seg.text.strip(), "start": seg.start, "end": seg.end} for seg in segments]


def extract_audio(video_path: str, audio_path: str):
    print(f"[INFO] Extracting audio from: {video_path}")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, logger=None)


def load_youtube_data(url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "format": "(bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4])[protocol^=http]",
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        print("[INFO] Done YDL")
        video_id = info["id"]
        title = info["title"]
        source_id = f"yt_{video_id}"
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(temp_dir, video_filename)

        audio_path = os.path.join(temp_dir, "audio.mp3")
        extract_audio(video_path, audio_path)

        transcript = transcribe_audio_whisper(audio_path)
        print(f"[DEBUG] Transcribed {len(transcript)} segments.")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps * 10))
        frames = []
        count, success = 0, True

        while success:
            success, image = cap.read()
            if success and count % frame_interval == 0:
                local_frame_path = os.path.join(temp_dir, f"frame_{count}.jpg")
                save_frame(image, local_frame_path)

                if os.path.exists(local_frame_path):
                    object_name = f"{source_id}/frame_{count}.jpg"
                    frames.append((count / fps, local_frame_path, object_name))
                else:
                    print(f"[WARN] Frame not saved: {local_frame_path}")
            count += 1
        cap.release()

        print(f"[DEBUG] Extracted {len(frames)} frames.")

        if not frames:
            raise HTTPException(status_code=500, detail="No frames extracted from the video.")

        embeddings = []
        for entry in transcript:
            text = entry["text"]
            start = entry["start"]
            print(f"[DEBUG] Processing segment: '{text[:30]}...' @ {start:.2f}s")

            closest_frame = min(frames, key=lambda f: abs(f[0] - start))
            local_path = closest_frame[1]
            object_name = closest_frame[2]

            if not os.path.exists(local_path):
                print(f"[ERROR] Frame file missing: {local_path}")
                continue

            try:
                image_url = upload_image(local_path, object_name)
                print(f"[DEBUG] Upload success: {image_url}")
                vec = embed_text_image(text, local_path)
                print(f"[DEBUG] Embedding shapes - text: {vec['text_embedding'].shape}, image: {vec['image_embedding'].shape}")

                metadata = {
                    "source_id": source_id,
                    "text": text,
                    "image_url": image_url,
                    "start_time": start,
                    "video_id": video_id,
                    "title": title,
                    "source": "youtube",
                }

                eid_text = add_embedding(vec["text_embedding"], metadata | {"modality": "text"})
                eid_image = add_embedding(vec["image_embedding"], metadata | {"modality": "image"})
                print(f"[DEBUG] Added embeddings: {eid_text}, {eid_image}")
                embeddings.extend([eid_text, eid_image])

            except Exception as e:
                print(f"[ERROR] Failed to embed segment @ {start:.2f}s: {e}")
                continue

            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)

        print(f"[INFO] Total valid chunks: {len(embeddings)}")
        save_source(source_id, f"s3://{source_id}/", title, embeddings)
        save_faiss_indices()

        return {"status": "ok", "title": title, "chunks": len(embeddings)}
