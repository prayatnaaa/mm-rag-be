import os
import cv2
import tempfile
import yt_dlp
import subprocess
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
from app.utils.file_utils import save_frame
from app.retriever.chromadb_index import add_embedding, embed_text, embed_image
from app.db.metadata_store import save_source
from app.utils.minio_client import upload_image, upload_video
import re

def clean_metadata(meta: dict):
    return {k: v for k, v in meta.items() if v is not None}

def is_logical_break(text):
    # Cek apakah teks diakhiri dengan titik atau simbol kalimat selesai
    return bool(re.search(r"[.!?…]['”\"]?\s*$", text.strip()))

def chunk_transcript(transcript, max_duration=30.0, max_chars=500):
    chunks = []
    current_chunk = []
    current_text = ""
    start_time = None
    end_time = None

    for seg in transcript:
        seg_text = seg["text"].strip()
        seg_start = seg["start"]
        seg_end = seg["end"]

        if not current_chunk:
            start_time = seg_start

        # Gabungkan teks untuk evaluasi panjang
        proposed_text = (current_text + " " + seg_text).strip() if current_text else seg_text
        proposed_duration = seg_end - start_time
        proposed_length = len(proposed_text)

        current_chunk.append(seg)
        current_text = proposed_text
        end_time = seg_end

        # Cek apakah harus dipotong di sini
        if (
            (proposed_length >= max_chars or proposed_duration >= max_duration)
            and is_logical_break(seg_text)
        ):
            chunks.append({
                "text": current_text.strip(),
                "start": start_time,
                "end": end_time,
            })
            current_chunk = []
            current_text = ""
            start_time = None
            end_time = None

    # Tambahkan sisa chunk terakhir
    if current_chunk:
        chunks.append({
            "text": current_text.strip(),
            "start": start_time,
            "end": end_time,
        })

    return chunks

def extract_audio(video_path: str, audio_path: str):
    print(f"[INFO] Extracting audio with ffmpeg: {video_path}")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "mp3", audio_path
    ]
    subprocess.run(cmd, check=True)

def extract_frames_every_n_seconds(video_path, output_dir, fps, interval=5):
    cap = cv2.VideoCapture(video_path)
    frame_map = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)

    for frame_no in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        success, frame = cap.read()
        if success:
            timestamp = frame_no / fps
            frame_path = os.path.join(output_dir, f"frame_{frame_no}.jpg")
            save_frame(frame, frame_path)
            frame_map[round(timestamp, 2)] = frame_path
        else:
            print(f"[WARN] Could not extract frame at {frame_no}.")
    cap.release()
    return frame_map

def transcribe_audio_whisper(audio_path: str):
    print(f"[INFO] Transcribing audio: {audio_path}")
    model = WhisperModel("base", compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    return [{"text": seg.text.strip(), "start": seg.start, "end": seg.end} for seg in segments]

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

        # --- Audio & Transkrip ---
        audio_path = os.path.join(temp_dir, "audio.mp3")
        extract_audio(video_path, audio_path)
        transcript = transcribe_audio_whisper(audio_path)
        chunks = chunk_transcript(transcript, max_duration=30.0)

        # --- Ambil frame setiap 2 detik ---
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        frame_map = extract_frames_every_n_seconds(video_path, temp_dir, fps, interval=2)

        # --- Embedding teks dan gambar ---
        embeddings = []

        for chunk in chunks:
            # --- TEXT EMBEDDING ---
            if chunk["start"] is None or chunk["end"] is None:
                print(f"[WARN] Skipping chunk with missing timestamps: {chunk}")
                continue
            text_vec = embed_text(chunk["text"])
            text_meta = {
                "source_id": source_id,
                "text": chunk["text"],
                "start_time": chunk["start"],
                "end_time": chunk["end"],
                "video_id": video_id,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                "title": title,
                "source": "youtube",
                "active": True,
            }
            eid_text = add_embedding(text_vec, clean_metadata(text_meta | {"modality": "text"}))
            embeddings.append(eid_text)

            # --- IMAGE EMBEDDING: frame dalam interval chunk ---
            for ts, frame_path in frame_map.items():
                if chunk["start"] <= ts <= chunk["end"]:
                    try:
                        frame_no = int(ts * fps)
                        image_object_name = f"{source_id}/frame_{frame_no}.jpg"
                        image_url = upload_image(frame_path, image_object_name)

                        if not image_url:
                            print(f"[WARN] Skipping image embedding at {ts} (upload_image returned None)")
                            continue

                        image_vec = embed_image(frame_path)
                        image_meta = {
                            "source_id": source_id,
                            "image_url": image_url,
                            "start_time": ts,
                            "end_time": ts,
                            "video_id": video_id,
                            "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                            "title": title,
                            "source": "youtube",
                            "active": True,
                        }
                        eid_image = add_embedding(image_vec, clean_metadata(image_meta | {"modality": "image"}))
                        embeddings.append(eid_image)

                    except Exception as e:
                        print(f"[ERROR] image embedding failed: {e}")

        # Cleanup
        os.remove(video_path)
        os.remove(audio_path)

        print(f"[INFO] Total embeddings: {len(embeddings)}")
        save_source(source_id, f"s3://{source_id}/", title, embeddings)

        return {"status": "ok", "title": title, "chunks": len(embeddings)}