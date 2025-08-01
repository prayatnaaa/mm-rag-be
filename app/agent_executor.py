import os
import hashlib
import json
import requests
from typing import List, Dict, Optional, Union
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from app.retriever.chromadb_index import search
from app.db.metadata_store import list_sources
# from deep_translator import GoogleTranslator
from langdetect import detect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DetectorFactory.seed = 0

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
vision_model = genai.GenerativeModel("gemini-2.0-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)

BASE_PROMPT = """
You are a highly capable expert multimodal assistant specializing in the analysis of visual and textual content from videos, including frames, screenshots, transcripts, and metadata. Your role is to synthesize information from retrieved contexts to deliver high-quality, human-level responses tailored to the user's query.

— Always prioritize clarity, factual accuracy, and relevance.
— Always directly and explicitly answer the user’s query. If the query asks for a specific fact (e.g., location, name, date), provide that fact clearly and concisely at the beginning of your response.
— If the query involves summarization or image analysis, never reference transcripts, timestamps, or raw metadata in the response. Abstract all context into fluent, natural language.
— If analyzing images, rely on a combination of visual content, embedded metadata, and any available captions. Provide a direct interpretation of what's visually present, including actions, objects, text in image, and scene layout.
— For summarization queries, combine all textual and visual insights into a coherent and concise summary. Do not mention the video structure (e.g., "this part of the video", "the transcript says", etc.).
— If the retrieved context does not contain the factual information required to answer the query, respond by stating that the information is not available or cannot be determined from the provided context.
— Respond in this language: **{lang}**
— Use polished, grammatically correct, well-structured sentences in {lang}.

**Query**: {query}

**Context**:
{contexts}
"""

def deduplicate_results(docs: List[str], metas: List[dict], dists: List[float]) -> List[Dict]:
    seen_hashes = set()
    deduped = []
    
    for doc, meta, dist in zip(docs, metas, dists):
        if not isinstance(meta, dict):
            logger.warning(f"Invalid metadata format: {meta}")
            meta = {"raw_meta": str(meta)}
        
        normalized_text = doc.strip().lower()
        text_hash = hashlib.md5(normalized_text.encode("utf-8")).hexdigest()
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            deduped.append({
                "text": doc,
                "metadata": meta,
                "distance": dist
            })
    
    return sorted(deduped, key=lambda x: x["distance"])[:10]

def build_prompt(query: str, contexts: List[Dict], lang: str = "en") -> str:
    context_str = ""
    print(lang)
    for i, ctx in enumerate(contexts):
        text = ctx.get("text", "(no text)")
        meta = ctx["metadata"]
        source = meta.get("source_id", "unknown")
        youtube_url = meta.get("youtube_url", "unknown")
        title = meta.get("title", "untitled")
        start_time = meta.get("start_time", "unknown")
        end_time = meta.get("end_time", "unknown")
        image_urls = meta.get("image_urls", "[]")
        captions = meta.get("captions", "[]")
        try:
            image_urls = json.loads(image_urls) if isinstance(image_urls, str) else image_urls
            captions = json.loads(captions) if isinstance(captions, str) else captions
            image_url = image_urls[0] if image_urls else "no image"
            caption = captions[0] if captions else "no caption"
        except Exception:
            image_url = "no image"
            caption = "no caption"
        
        context_str += (
            f"- [{i+1}] Title: {title}\n"
            f"- [{i+1}] Text: {text.strip()[:200]}...\n"
            f"  Source: {source}, YouTube URL: {youtube_url}\n"
            f"  Time: {start_time}s - {end_time}s\n"
            f"  Image: {image_url}, Caption: {caption}\n"
        )
    
    return BASE_PROMPT.format(query=query, contexts=context_str or "None", lang=lang)

def generate_image_description(image: Optional[Union[Image.Image, bytes]], contexts: List[Dict], query: str, lang: str = "en") -> Dict:
    parts = [f"Describe the visual content related to '{query}' based on the provided contexts and images."]
    
    for ctx in contexts:
        text = ctx.get("text", "")
        meta = ctx["metadata"]
        source = meta.get("source_id", "unknown")
        youtube_url = meta.get("youtube_url", "unknown")
        timestamp = meta.get("start_time", "unknown")
        captions = meta.get("captions", "[]")
        try:
            captions = json.loads(captions) if isinstance(captions, str) else captions
            caption = captions[0] if captions else "no caption"
        except Exception:
            caption = "no caption"
        parts.append(
            f"[Context] Source: {source}, YouTube URL: {youtube_url}, "
            f"Timestamp: {timestamp}s, Text: {text[:200]}..., Caption: {caption}"
        )
    
    image_parts = []
    if image:
        if isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
        else:
            image_bytes = image
        image_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
    
    for ctx in contexts[:2]:
        image_urls = ctx["metadata"].get("image_urls", "[]")
        print(image_urls)
        try:
            image_urls = json.loads(image_urls) if isinstance(image_urls, str) else image_urls
            if image_urls:
                for url in image_urls[:1]: 
                    try:
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            image_parts.append({"mime_type": "image/jpeg", "data": response.content})
                            parts.append(f"[Image Context] Fetched image from {url}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch image from {url}: {str(e)}")
        except Exception:
            logger.warning(f"Invalid image_urls format in context: {image_urls}")
    
    try:
        if image_parts:
            response = vision_model.generate_content(parts + image_parts)
        else:
            response = vision_model.generate_content(parts)
        answer = response.text
        return {"answer": answer, "evidence": contexts}
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        error_msg = f"Error analyzing image: {str(e)}"
        return {"answer": error_msg, "evidence": contexts}

State = Dict[str, Union[str, List[Dict], Optional[Union[str, Image.Image]], str]]

def classify_query(state: State) -> State:
    query = state["query"]
    image = state.get("image")
    
    lang = detect(query)
    print(f"GOOGLE TRANSLATOR: {lang}")
    
    
    classifier_prompt = f"""Classify the user query into one of the following:
        - "summarize_all": if asking for a full overview or summary of all sources.
        - "image_description": if asking about visual content (e.g., "describe the image", "baju apa", "what is in the video") or image-only input.
        - "answer_with_context": if asking a specific question or summary of a specific source.
        - "other": if the query is irrelevant to the context (e.g. personal questions about unrelated figures, open-domain unrelated topics).


        Query: "{query}"
        Image Provided: {bool(image)}
        Answer only with the label."""
    
    result = llm.invoke(classifier_prompt).content.strip().lower()
    label = result if result in {"summarize_all", "image_description", "answer_with_context", "other"} else "other"
    return {**state, "query_type": label, "lang": lang}

def retrieve_contexts(state: State) -> State:
    """
    Retrieve contexts using the search function, prioritizing visual content for image-related queries.
    """
    query = state["query"]
    image = state.get("image")
    
    if state["query_type"] == "image_description" and query:
        query = f"{query}"

    if state["query_type"] == "summarize_all":
        sources = list_sources()
        source_ids = [s["source_id"] for s in sources]
        contexts = []
        for source_id in source_ids:
            results = search(
                query=query,
                image=image,
                n_results=30,
                where={"source_id": {"$eq": source_id}}
            )
            contexts.extend(deduplicate_results(
                results["results"]["documents"][0],
                results["results"]["metadatas"][0],
                results["results"]["distances"][0]
            ))
    else:
        results = search(query=query, image=image, n_results=50)
        contexts = deduplicate_results(
            results["results"]["documents"][0],
            results["results"]["metadatas"][0],
            results["results"]["distances"][0]
        )
    
    return {**state, "contexts": contexts}

def summarize_all(state: State) -> State:
    """
    Summarize all available contexts.
    """
    contexts = state.get("contexts", [])
    lang = state.get("lang")

    for ctx in contexts:
        ctx["metadata"]["type"] = "summarize"
    
    prompt = build_prompt("Provide a comprehensive summary of all available information from all sources.", contexts, lang)
    result = llm.invoke(prompt).content
    
    return {**state, "answer": result, "contexts": contexts}

def describe_image(state: State) -> State:
    """
    Describe visual content based on image or contexts.
    """
    query = state["query"]
    image = state.get("image")
    contexts = state.get("contexts", [])
    lang = state.get("lang")

    for ctx in contexts:
        ctx["metadata"]["type"] = "describe_image"
    
    result = generate_image_description(image, contexts, query, lang)
    return {**state, "answer": result["answer"], "contexts": contexts}

def answer_with_context(state: State) -> State:
    query = state["query"]
    contexts = state.get("contexts", [])
    lang = state.get("lang")

    for ctx in contexts:
        ctx["metadata"]["type"] = "answer_with_context"
    
    prompt = build_prompt(query, contexts, lang)
    result = llm.invoke(prompt).content
    
    return {**state, "answer": result, "contexts": contexts}

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
            error_msg = "Sorry, no answer could be found."
            return {
                "query": query,
                "answer": error_msg,
                "contexts": []
            }
        
        return {
            "query": query,
            "answer": state["answer"],
            "contexts": state.get("contexts", [])
        }
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        return {
            "query": query,
            "answer": error_msg,
            "contexts": []
        }