from fastapi import HTTPException
# from pytube import YouTube
from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import os, cv2
from app.utils.file_utils import save_frame
from app.retriever.embed_clip import embed_text_image
from app.retriever.faiss_index import add_embedding
from app.db.metadata_store import save_source
from app.utils.minio_client import upload_image

def load_youtube_data(url: str):
    yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
    video_id = yt.video_id
    title = yt.title
    print("yt video id ", video_id)
    print("yt video title ", title)

    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript_obj = transcripts.find_generated_transcript(['id']) 
        transcript = transcript_obj.fetch()
        print("masuk")
        print(transcript)
    except Exception as e:
        print("ppp")
        print("Error:", e)
        raise HTTPException(status_code=400, detail="No transcript found for this video.")

    # Download video
    video_path = yt.streams.filter(file_extension='mp4').first().download(filename=f"{video_id}.mp4")
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 10)  # 1 frame every 10 seconds

    success, image, count = True, None, 0
    source_id = f"yt_{video_id}"
    os.makedirs(f"storage/temp/{source_id}", exist_ok=True)

    frames = []
    while success:
        success, image = cap.read()
        if success and count % frame_interval == 0:
            path = f"storage/temp/{source_id}/frame_{count}.jpg"
            save_frame(image, path)
            frames.append(path)
        count += 1
    cap.release()

    embeddings = []
    for entry in transcript:
        print(entry)
        text = entry.text
        for img_path in frames:
            img_name = os.path.basename(img_path)
            url = upload_image(img_path, f"{source_id}/{img_name}")
            vec = embed_text_image(text, img_path)
            eid = add_embedding(vec, metadata={"image_url": url, "text": text})
            embeddings.append(eid)
            os.remove(img_path)

    save_source(source_id, url, title, embeddings)
    return {"status": "ok", "title": title, "chunks": len(embeddings)}

