import os
import cv2
import tempfile
import yt_dlp
import subprocess
from faster_whisper import WhisperModel
from app.utils.file_utils import save_frame
from app.retriever.chromadb_index import add_embedding, embed_text, embed_image
from app.db.metadata_store import save_source
from app.utils.minio_client import upload_image, upload_video
import re
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()
whisper_model = WhisperModel("base", compute_type="int8")

def clean_metadata(meta: dict):
    return {k: v for k, v in meta.items() if v is not None}

def is_logical_break(text):
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

        proposed_text = (current_text + " " + seg_text).strip() if current_text else seg_text
        proposed_duration = seg_end - start_time
        proposed_length = len(proposed_text)

        current_chunk.append(seg)
        current_text = proposed_text
        end_time = seg_end

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

    if current_chunk:
        chunks.append({
            "text": current_text.strip(),
            "start": start_time,
            "end": end_time,
        })

    return chunks

def extract_audio(video_path: str, audio_path: str):
    logger.info(f"Extracting audio with ffmpeg: {video_path}")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "mp3", audio_path
    ]
    subprocess.run(cmd, check=True)

def extract_frames_every_n_seconds(video_path, output_dir, fps, interval=10):
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
            logger.warning(f"Could not extract frame at {frame_no}.")
    cap.release()
    return frame_map

def transcribe_audio_whisper(audio_path: str):
    logger.info(f"Transcribing audio: {audio_path}")
    segments, _ = whisper_model.transcribe(audio_path)
    return [{"text": seg.text.strip(), "start": seg.start, "end": seg.end} for seg in segments]

def load_youtube_data(url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts = {
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "format": "(bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4])[protocol^=http]",
            "quiet": True,
        }
        try:
            logger.info(f"Downloading YouTube video: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

            video_id = info["id"]
            title = info["title"]
            source_id = f"yt_{video_id}"
            video_filename = f"{video_id}.mp4"
            video_path = os.path.join(temp_dir, video_filename)
            object_name = f"videos/{video_filename}"

            logger.info(f"Uploading video to MinIO: {object_name}")
            upload_video(video_path, object_name)

            audio_path = os.path.join(temp_dir, "audio.mp3")
            extract_audio(video_path, audio_path)
            transcript = transcribe_audio_whisper(audio_path)
            chunks = chunk_transcript(transcript, max_duration=30.0, max_chars=500)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            logger.info("Extracting frames from video")
            frame_map = extract_frames_every_n_seconds(video_path, temp_dir, fps, interval=10)

            results = []

            for chunk in chunks:
                start = chunk["start"]
                end = chunk["end"]
                chunk_text = chunk["text"]

                captions = []
                image_urls = []
                image_embeddings = []

                frame_times = [t for t in frame_map.keys() if start <= t <= end]
                frame_paths = [frame_map[t] for t in frame_times]

                if frame_paths:
                    logger.info(f"Generating captions for {len(frame_paths)} frames in chunk {start}-{end}")
                    try:
                        images = [Image.open(path).convert("RGB") for path in frame_paths]
                        inputs = processor(images, return_tensors="pt")
                        with torch.no_grad():
                            outputs = model.generate(**inputs)
                        captions = [processor.decode(out, skip_special_tokens=True) for out in outputs]
                    except Exception as e:
                        logger.error(f"Failed to generate captions for chunk {start}-{end}: {str(e)}")
                        captions = ["" for _ in frame_paths]

                    for idx, frame_path in enumerate(frame_paths):
                        try:
                            logger.debug(f"Embedding image: {frame_path}")
                            image_embedding = embed_image(frame_path)  
                            image_embeddings.append(image_embedding)

                            frame_no = int(frame_times[idx] * fps)
                            image_object_name = f"{source_id}/frame_{frame_no}.jpg"
                            image_url = upload_image(frame_path, image_object_name)
                            if image_url:
                                image_urls.append(image_url)
                            else:
                                logger.warning(f"Failed to upload image: {image_object_name}")
                        except Exception as e:
                            logger.error(f"Failed to process frame {frame_path}: {str(e)}")
                            continue

                combined_text = chunk_text + " " + " ".join(captions) if captions else chunk_text
                logger.debug(f"Embedding text for chunk {start}-{end}")
                text_vec = embed_text(combined_text)  

                meta = {
                    "source_id": source_id,
                    "video_id": video_id,
                    "title": title,
                    "text": json.dumps(combined_text),
                    "captions": json.dumps(captions),
                    "image_urls": json.dumps(image_urls),
                    "image_embeddings": json.dumps([emb.tolist() for emb in image_embeddings]),  
                    "start_time": chunk["start"],
                    "end_time": chunk["end"],
                    "youtube_url": f"https://youtube.com/watch?v={video_id}",
                    "modality": "multimodal",
                    "source": "youtube",
                    "active": True,
                }

                logger.info(f"Adding text embedding for chunk {start}-{end}")
                add_embedding(text_vec, clean_metadata(meta))

                for idx, img_embedding in enumerate(image_embeddings):
                    image_meta = meta.copy()
                    image_meta["frame_no"] = int(frame_times[idx] * fps) if idx < len(frame_times) else None
                    image_meta["image_url"] = image_urls[idx] if idx < len(image_urls) else None
                    image_meta["modality"] = "image"
                    logger.debug(f"Adding image embedding for frame {image_meta['frame_no']}")
                    add_embedding(img_embedding, clean_metadata(image_meta))

                results.append(meta)

            logger.info(f"Saving source metadata for {source_id}")
            save_source(source_id, f"s3://{source_id}/", title, results)

            return results

        except Exception as e:
            logger.error(f"Failed to process YouTube video {url}: {str(e)}")
            return []