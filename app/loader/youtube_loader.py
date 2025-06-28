import os
import cv2
import tempfile
import yt_dlp
import subprocess
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip
from app.utils.file_utils import save_frame
from app.retriever.chromadb_index import add_embedding, embed_text, embed_image
from app.db.metadata_store import save_source
from app.utils.minio_client import upload_image, upload_video
import concurrent.futures
import re

def chunk_transcript(transcript, max_duration=30.0, max_chars=400):
    chunks = []
    current_chunk = []
    current_text = ""
    start_time = None
    end_time = None

    def is_logical_break(text):
        return bool(re.search(r"[.!?…](['”\"])?\s*$", text.strip()))

    for seg in transcript:
        seg_text = seg["text"].strip()

        if not current_chunk:
            start_time = seg["start"]

        proposed_text = (current_text + " " + seg_text).strip() if current_text else seg_text
        proposed_duration = seg["end"] - start_time
        proposed_length = len(proposed_text)

        # Logika utama pemotongan chunk:
        should_break = (
            (proposed_length > max_chars or proposed_duration > max_duration)
            and is_logical_break(current_text)
        )

        if should_break:
            chunks.append({
                "text": current_text.strip(),
                "start": start_time,
                "end": end_time,
            })
            current_chunk = [seg]
            current_text = seg_text
            start_time = seg["start"]
        else:
            current_chunk.append(seg)
            current_text = proposed_text

        end_time = seg["end"]

    # Tambahkan sisa chunk terakhir
    if current_text:
        chunks.append({
            "text": current_text.strip(),
            "start": start_time,
            "end": end_time,
        })

    return chunks

def transcribe_audio_whisper(audio_path: str):
    print(f"[INFO] Transcribing audio: {audio_path}")
    model = WhisperModel("base", compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    return [{"text": seg.text.strip(), "start": seg.start, "end": seg.end} for seg in segments]

def extract_audio(video_path: str, audio_path: str):
    print(f"[INFO] Extracting audio with ffmpeg: {video_path}")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "mp3", audio_path
    ]
    subprocess.run(cmd, check=True)

def load_youtube_data(url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        # --- Download YouTube Video ---
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

        # --- Extract Audio & Transcribe ---
        audio_path = os.path.join(temp_dir, "audio.mp3")
        extract_audio(video_path, audio_path)
        transcript = transcribe_audio_whisper(audio_path)
        print(f"[DEBUG] Transcribed {len(transcript)} segments.")

        # --- Chunk Transcripts (~20s each) ---
        chunks = chunk_transcript(transcript, max_duration=20.0)

        # --- Ambil fps satu kali ---
        fps_cap = cv2.VideoCapture(video_path)
        fps = fps_cap.get(cv2.CAP_PROP_FPS)
        fps_cap.release()

        # --- Proses tiap chunk secara paralel ---
        def process_chunk(chunk):
            start = chunk["start"]
            end = chunk["end"]
            midpoint = (start + end) / 2.0
            frame_number = int(fps * midpoint)

            cap = None
            frame_path = None

            try:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, image = cap.read()
                cap.release()

                if not success:
                    return None

                frame_path = os.path.join(temp_dir, f"frame_{frame_number}.jpg")
                save_frame(image, frame_path)

                image_object_name = f"{source_id}/frame_{frame_number}.jpg"
                image_url = upload_image(frame_path, image_object_name)

                text_vec = embed_text(chunk["text"])
                image_vec = embed_image(frame_path)

                metadata = {
                    "source_id": source_id,
                    "text": chunk["text"],
                    "image_url": image_url,
                    "start_time": start,
                    "end_time": end,
                    "video_id": video_id,
                    "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": title,
                    "source": "youtube",
                    "active": True,
                }

                eid_text = add_embedding(text_vec, metadata | {"modality": "text"})
                eid_image = add_embedding(image_vec, metadata | {"modality": "image"})

                return [eid_text, eid_image]

            except Exception as e:
                print(f"[ERROR] Failed to process chunk: {e}")
                return None

            finally:
                if cap:
                    cap.release()
                if frame_path and os.path.exists(frame_path):
                    os.remove(frame_path)

        embeddings = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_chunk, chunks))
            for result in results:
                if result:
                    embeddings.extend(result)

        # Cleanup
        os.remove(video_path)
        os.remove(audio_path)

        print(f"[INFO] Total valid chunks: {len(embeddings)}")
        save_source(source_id, f"s3://{source_id}/", title, embeddings)

        return {"status": "ok", "title": title, "chunks": len(embeddings)}