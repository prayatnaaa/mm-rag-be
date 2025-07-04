import os
import hashlib
import requests
import re
from typing import List, Dict, Optional, Union
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from app.retriever.chromadb_index import search
from app.db.metadata_store import list_sources
from app.datastore.model import create_new_chat, save_chat_contents
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
vision_model = genai.GenerativeModel("gemini-2.0-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)

BASE_PROMPT = """You are an expert multimodal assistant capable of analyzing text and images from video transcripts and frames. Below are retrieved text snippets and image references relevant to the user's query.

Your task is to provide a clear, concise, and accurate answer based on the provided context. For image-related queries, describe visual content using image metadata and direct image analysis. For summarization, include insights from all available sources. Respond in the user's language.

Please format your response using **Markdown syntax compatible with React Markdown**, including headings, lists, and code blocks where appropriate.

[QUESTION]
{query}

[CONTEXT]
Text Contexts:
{text_contexts}

Image Contexts:
{image_contexts}
"""

def parse_time_from_query(query: str, lang: str = "en") -> Optional[float]:
    """
    Parse time references from the query (e.g., 'menit ke-5', '6 minutos', '120 secondes') and return time in seconds.
    Returns None if no time reference is found.
    """
    patterns = {
        "id": [
            r"(?:menit\s*ke-|ke\s*menit\s*)(\d+)",  # e.g., "menit ke-5"
            r"(\d+)\s*menit",                      # e.g., "6 menit"
            r"(?:detik\s*ke-|ke\s*detik\s*)(\d+)", # e.g., "detik ke-120"
            r"(\d+)\s*detik"                       # e.g., "120 detik"
        ],
        "en": [
            r"(?:minute\s*|minutes\s*)(\d+)",      # e.g., "6 minutes"
            r"(?:second\s*|seconds\s*)(\d+)"       # e.g., "120 seconds"
        ],
        "es": [
            r"(?:minuto\s*|minutos\s*)(\d+)",      # e.g., "6 minutos"
            r"(?:segundo\s*|segundos\s*)(\d+)"     # e.g., "120 segundos"
        ],
        "fr": [
            r"(?:minute\s*|minutes\s*)(\d+)",      # e.g., "6 minutes"
            r"(?:seconde\s*|secondes\s*)(\d+)"     # e.g., "120 secondes"
        ]
    }
    
    lang_patterns = patterns.get(lang, patterns["en"])
    for pattern in lang_patterns:
        match = re.search(pattern, query.lower())
        if match:
            value = float(match.group(1))
            if "menit" in pattern or "minute" in pattern or "minuto" in pattern:
                return value * 60  
            return value
    
    return None

def deduplicate_results(docs: List[str], metas: List[dict], dists: List[float]) -> List[Dict]:
    seen_hashes = set()
    deduped = []
    
    for doc, meta, dist in zip(docs, metas, dists):
        if isinstance(meta, dict):
            clean_meta = meta
        elif isinstance(meta, list) and meta and isinstance(meta[0], dict):
            clean_meta = meta[0]
        else:
            clean_meta = {"raw_meta": str(meta)}
        
        normalized_text = doc.strip().lower()
        text_hash = hashlib.md5(normalized_text.encode("utf-8")).hexdigest()
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            deduped.append({
                "text": doc,
                "metadata": clean_meta,
                "distance": dist
            })
    
    return sorted(deduped, key=lambda x: x["distance"])[:10]

def build_prompt(query: str, text_contexts: List[Dict], image_contexts: List[Dict], lang: str = "en") -> str:
    text_context_str = ""
    image_context_str = ""
    
    for i, ctx in enumerate(text_contexts):
        text = ctx.get("text", "(no text)")
        meta = ctx["metadata"]
        source = meta.get("source_id", "unknown")
        youtube_url = meta.get("youtube_url", "unknown")
        start_time = meta.get("start_time", "unknown")
        end_time = meta.get("end_time", "unknown")
        text_context_str += f"- [{i+1}] Text: {text.strip()[:200]}...\n  Source: {source}, YouTube URL: {youtube_url}, Time: {start_time}s - {end_time}s\n"
    
    for i, ctx in enumerate(image_contexts):
        meta = ctx["metadata"]
        source = meta.get("source_id", "unknown")
        youtube_url = meta.get("youtube_url", "unknown")
        image_url = meta.get("image_url", "no image")
        timestamp = meta.get("start_time", "unknown")
        image_context_str += f"- [{i+1}] Image: {image_url}\n  Source: {source}, YouTube URL: {youtube_url}, Timestamp: {timestamp}s\n"
    
    prompt = BASE_PROMPT.format(query=query, text_contexts=text_context_str or "None", image_contexts=image_context_str or "None")
    # Translate prompt to the detected language if not English
    if lang != "en":
        try:
            prompt = GoogleTranslator(source="en", target=lang).translate(prompt)
        except Exception as e:
            print(f"[Translation Error] Failed to translate prompt to {lang}: {str(e)}")
    
    return prompt

def generate_image_description(image: Optional[Union[Image.Image, bytes]], contexts: List[Dict], query: str, lang: str = "en") -> Dict:
    parts = [f"Describe the visual content related to {query or 'the video frames'} based on the following contexts."]
    if lang != "en":
        parts[0] = GoogleTranslator(source="en", target=lang).translate(parts[0])
    
    for ctx in contexts:
        text = ctx.get("text", "")
        meta = ctx["metadata"]
        source = meta.get("source_id", "unknown")
        youtube_url = meta.get("youtube_url", "unknown")
        timestamp = meta.get("start_time", "unknown")
        if text:
            parts.append(f"[Video Context] Source: {source}, YouTube URL: {youtube_url}, Timestamp: {timestamp}s, Text: {text[:200]}...")
    
    try:
        if isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            response = vision_model.generate_content(parts + [{"mime_type": "image/jpeg", "data": image_bytes}])
            answer = response.text
        elif image:
            response = vision_model.generate_content(parts + [{"mime_type": "image/jpeg", "data": image}])
            answer = response.text
        else:
            # Try fetching image from image_contexts
            for ctx in [c for c in contexts if c["metadata"].get("image_url")]:
                image_url = ctx["metadata"].get("image_url")
                try:
                    response = requests.get(image_url, timeout=5)
                    if response.status_code == 200:
                        image_bytes = response.content
                        parts.append(f"[Image Context] Fetched image from {image_url}")
                        response = vision_model.generate_content(parts + [{"mime_type": "image/jpeg", "data": image_bytes}])
                        answer = response.text
                        break
                except Exception as e:
                    parts.append(f"[Warning] Failed to fetch image from {image_url}: {str(e)}")
            else:
                # Fallback: Infer from text contexts and metadata
                response = vision_model.generate_content(parts)
                answer = response.text
        
        # Translate response to the detected language if not English
        if lang != "en":
            try:
                answer = GoogleTranslator(source="en", target=lang).translate(answer)
            except Exception as e:
                print(f"[Translation Error] Failed to translate answer to {lang}: {str(e)}")
        
        return {"answer": answer, "evidence": contexts}
    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        if lang != "en":
            try:
                error_msg = GoogleTranslator(source="en", target=lang).translate(error_msg)
            except Exception:
                error_msg = f"Kesalahan saat menganalisis gambar: {str(e)}"  # Fallback to Indonesian
        return {"answer": error_msg, "evidence": contexts}

# ==== LangGraph Nodes ====

State = Dict[str, Union[str, List[Dict], Optional[Union[str, Image.Image]], str]]

def classify_query(state: State) -> State:
    query = state["query"]
    image = state.get("image")
    
    # Detect language dynamically
    try:
        lang = detect(query) if query else "en"
    except Exception:
        lang = "en" 
    
    classifier_prompt = f"""Classify the user query into one of the following:
- "summarize_all": if asking for a full overview or summary of all available data from all sources.
- "image_description": if asking about visual content or image-only input (e.g., "describe the image", "jelaskan gambar")
- "answer_with_context": if asking a specific question needing retrieval also if if asking a specific summary of spesific source.
- "other": if it doesn't fit the above categories

Query: "{query}"
Image Provided: {bool(image)}
Answer only with the label."""
    
    result = llm.invoke(classifier_prompt).content.strip().lower()
    label = result if result in {"summarize_all", "image_description", "answer_with_context", "other"} else "other"
    return {**state, "query_type": label, "lang": lang}

def retrieve_contexts(state: State) -> State:
    query = state["query"]
    image = state.get("image")
    
    if state["query_type"] == "summarize_all":
        sources = list_sources()
        source_ids = [s["source_id"] for s in sources]
        text_results = []
        image_results = []
        for source_id in source_ids:
            results = search(query=query, image=image, n_results=10, where={"source_id": {"$eq": source_id}})
            text_results.extend(deduplicate_results(
                results["text_results"]["documents"][0],
                results["text_results"]["metadatas"][0],
                results["text_results"]["distances"][0]
            ))
            image_results.extend(deduplicate_results(
                results["image_results"]["documents"][0],
                results["image_results"]["metadatas"][0],
                results["image_results"]["distances"][0]
            ))
    else:
        results = search(query=query, image=image, n_results=50)
        text_results = deduplicate_results(
            results["text_results"]["documents"][0],
            results["text_results"]["metadatas"][0],
            results["text_results"]["distances"][0]
        )
        image_results = deduplicate_results(
            results["image_results"]["documents"][0],
            results["image_results"]["metadatas"][0],
            results["image_results"]["distances"][0]
        )
    
    return {
        **state,
        "text_contexts": text_results,
        "image_contexts": image_results
    }

def summarize_all(state: State) -> State:
    text_contexts = state.get("text_contexts", [])
    image_contexts = state.get("image_contexts", [])
    lang = state.get("lang", "en")
    
    prompt = build_prompt("Provide a comprehensive summary of all available information from all sources, combining text and visual insights.", text_contexts, image_contexts, lang)
    result = llm.invoke(prompt).content
    
    # Translate result to the detected language if not English
    if lang != "en":
        try:
            result = GoogleTranslator(source="en", target=lang).translate(result)
        except Exception as e:
            print(f"[Translation Error] Failed to translate summary to {lang}: {str(e)}")
    
    return {
        **state,
        "answer": result,
        "text_contexts": text_contexts,
        "image_contexts": image_contexts
    }

def describe_image(state: State) -> State:
    query = state["query"]
    image = state.get("image")
    image_contexts = state.get("image_contexts", [])
    text_contexts = state.get("text_contexts", [])
    lang = state.get("lang", "en")
    
    if image:
        result = generate_image_description(image, image_contexts + text_contexts, query, lang)
        return {
            **state,
            "answer": result["answer"],
            "text_contexts": text_contexts,
            "image_contexts": image_contexts
        }
    
    target_time = parse_time_from_query(query, lang)
    if target_time:
        time_window = 30  # Configurable time window in seconds
        image_contexts = sorted(
            [ctx for ctx in image_contexts if abs(float(ctx["metadata"].get("start_time", 0)) - target_time) <= time_window],
            key=lambda x: abs(float(x["metadata"].get("start_time", 0)) - target_time)
        )[:1] 
    
    result = generate_image_description(None, image_contexts + text_contexts, query, lang)
    return {
        **state,
        "answer": result["answer"],
        "text_contexts": text_contexts,
        "image_contexts": image_contexts
    }

def answer_with_context(state: State) -> State:
    query = state["query"]
    text_contexts = state.get("text_contexts", [])
    image_contexts = state.get("image_contexts", [])
    lang = state.get("lang", "en")
    
    prompt = build_prompt(query, text_contexts, image_contexts, lang)
    result = llm.invoke(prompt).content
    
    # Translate result to the detected language if not English
    if lang != "en":
        try:
            result = GoogleTranslator(source="en", target=lang).translate(result)
        except Exception as e:
            print(f"[Translation Error] Failed to translate answer to {lang}: {str(e)}")
    
    return {
        **state,
        "answer": result,
        "text_contexts": text_contexts,
        "image_contexts": image_contexts
    }


def build_graph():
    workflow = StateGraph(State)
    
    workflow.add_node("classify", classify_query)
    workflow.add_node("retrieve", retrieve_contexts)
    workflow.add_node("summarize_all", summarize_all)
    workflow.add_node("describe_image", describe_image)
    workflow.add_node("answer_with_context", answer_with_context)
    
    workflow.set_entry_point("classify")
    workflow.add_conditional_edges("classify", lambda s: s["query_type"], {
        "summarize_all": "retrieve",
        "image_description": "retrieve",
        "answer_with_context": "retrieve",
        "other": END
    })
    
    workflow.add_conditional_edges("retrieve", lambda s: s["query_type"], {
        "summarize_all": "summarize_all",
        "image_description": "describe_image",
        "answer_with_context": "answer_with_context",
        "other": END
    })
    
    workflow.add_edge("summarize_all", END)
    workflow.add_edge("describe_image", END)
    workflow.add_edge("answer_with_context", END)
    
    return workflow.compile()

# ==== Public Function ====

graph = build_graph()

def run_agentic_rag(query: str = "", image: Optional[Union[str, Image.Image]] = None) -> Dict:
    try:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        state = graph.invoke({
            "query": query,
            "image": image
        })
        
        if not state or "answer" not in state:
            lang = detect(query) if query else "en"
            error_msg = "Sorry, no answer could be found."
            if lang != "en":
                try:
                    error_msg = GoogleTranslator(source="en", target=lang).translate(error_msg)
                except Exception:
                    error_msg = "Maaf, tidak ada jawaban yang bisa ditemukan."  # Fallback to Indonesian
            return {
                "query": query,
                "answer": error_msg,
                "text_contexts": [],
                "image_contexts": []
            }
        
        chat_id = create_new_chat(topic=query or "Image-based query")
        if chat_id:
            save_chat_contents(
                chat_id=chat_id,
                query=query or "Image-based query",
                answer=state["answer"],
                used_contexts=state.get("text_contexts", []) + state.get("image_contexts", [])
            )
        
        return {
            "query": query,
            "answer": state["answer"],
            "text_contexts": state.get("text_contexts", []),
            "image_contexts": state.get("image_contexts", [])
        }
    
    except Exception as e:
        lang = detect(query) if query else "en"
        error_msg = f"Error: {str(e)}"
        if lang != "en":
            try:
                error_msg = GoogleTranslator(source="en", target=lang).translate(error_msg)
            except Exception:
                error_msg = f"Kesalahan: {str(e)}"  # Fallback to Indonesian
        return {
            "query": query,
            "answer": error_msg,
            "text_contexts": [],
            "image_contexts": []
        }