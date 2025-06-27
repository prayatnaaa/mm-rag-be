import os
import cv2
import tempfile
import yt_dlp
from fastapi import HTTPException
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
from app.utils.file_utils import save_frame
from app.retriever.embed_clip import embed_text_image
from app.retriever.chromadb_index import add_embedding
from app.db.metadata_store import save_source
from app.utils.minio_client import upload_image, upload_video
import concurrent.futures


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

        video_id = info["id"]
        title = info["title"]
        source_id = f"yt_{video_id}"
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(temp_dir, video_filename)

        object_name = f"videos/{video_filename}"
        upload_video(video_path, object_name)

        audio_path = os.path.join(temp_dir, "audio.mp3")
        extract_audio(video_path, audio_path)

        transcript = transcribe_audio_whisper(audio_path)
        print(f"[DEBUG] Transcribed {len(transcript)} segments.")

        cap = cv2.VideoCapture(os.path.join(temp_dir, video_filename))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_map = {} 

        for entry in transcript:
            start = entry["start"]
            frame_number = int(fps * start)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = cap.read()
            if success:
                frame_path = os.path.join(temp_dir, f"frame_{frame_number}.jpg")
                save_frame(image, frame_path)
                frame_map[start] = frame_path
            else:
                print(f"[WARN] Failed to read frame at {start:.2f}s")

        cap.release()
        os.remove(video_path)
        os.remove(audio_path)

        print(f"[DEBUG] Extracted {len(frame_map)} frames.")

        if not frame_map:
            raise HTTPException(status_code=500, detail="No frames extracted.")

        def process_entry(entry):
            start = entry["start"]
            text = entry["text"]
            local_path = frame_map.get(start)
            if not local_path or not os.path.exists(local_path):
                return None

            try:
                object_name = f"{source_id}/frame_{int(fps * start)}.jpg"
                image_url = upload_image(local_path, object_name)
                vec = embed_text_image(text, local_path)

                metadata = {
                    "source_id": source_id,
                    "text": text,
                    "image_url": image_url,
                    "start_time": start,
                    "end_time": entry["end"],
                    "video_id": video_id,
                    "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": title,
                    "source": "youtube",
                }

                eid_text = add_embedding(vec["text_embedding"], metadata | {"modality": "text"})
                eid_image = add_embedding(vec["image_embedding"], metadata | {"modality": "image"})

                return [eid_text, eid_image]
            except Exception as e:
                return None
            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)

        embeddings = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_entry, transcript))
            for result in results:
                if result:
                    embeddings.extend(result)

        print(f"[INFO] Total valid chunks: {len(embeddings)}")
        save_source(source_id, f"s3://{source_id}/", title, embeddings)

        return {"status": "ok", "title": title, "chunks": len(embeddings)}
