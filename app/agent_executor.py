import os
import hashlib
from typing import List, Dict, Optional, Union
from PIL import Image
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable, RunnableMap, RunnableLambda
from langgraph.graph import StateGraph, END

from app.datastore.model import create_new_chat, save_chat_contents
from app.retriever.chromadb_index import search

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)

BASE_PROMPT = """You are an expert assistant. Below are information chunks retrieved from multiple video transcripts and image links that may help answer the user's question.

Your task is to analyze these snippets and image references to provide a clear, concise, and informative answer to the user. Consider both text and image context. If image URLs are included, assume that the image is relevant and try to infer its meaning from the text and surrounding context. Prioritize accurate and reasoned answers over speculation. If multiple answers are possible, summarize or compare them.

Always respond in the same language as the user's question.

[QUESTION]
{query}
"""

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

    return deduped

def build_prompt(query: str, contexts: List[Dict]) -> str:
    prompt = BASE_PROMPT.format(query=query)

    for i, ctx in enumerate(contexts):
        text = ctx.get("text", "(no text)")
        meta = ctx["metadata"]
        source = meta.get("source_id", "unknown")
        image_url = meta.get("image_url")

        prompt += f"\n[CONTEXT {i+1}]\n"
        prompt += f"- Text: {text.strip()[:200]}...\n"
        prompt += f"- Source: {source}\n"
        if image_url:
            prompt += f"- Image URL: {image_url}\n"

    return prompt

# ==== LangGraph Nodes ====

State = dict

def classify_query(state: State) -> State:
    query = state["query"]
    classifier_prompt = f"""Classify the following user query into one of the following:
- "summarize_all": if asking for full overview, summary, or all data
- "answer_with_context": if asking a specific question needing retrieval
- "other": if it's something else

Query: "{query}"
Answer only with the label."""
    
    result = llm.invoke(classifier_prompt).content.strip().lower()
    label = result if result in {"summarize_all", "answer_with_context"} else "other"
    return {
        **state,
        "query_type": label
    }

def retrieve_contexts(state: State) -> State:
    query = state["query"]
    image = state.get("image")

    results = search(query=query, image=image, n_results=50)

    # Top-N from text & image
    n_chunks = 10
    text = sorted(
        zip(results["text_results"]["documents"][0], results["text_results"]["metadatas"][0], results["text_results"]["distances"][0]),
        key=lambda x: x[2]
    )[:n_chunks]

    image = sorted(
        zip(results["image_results"]["documents"][0], results["image_results"]["metadatas"][0], results["image_results"]["distances"][0]),
        key=lambda x: x[2]
    )[:n_chunks]

    all_docs = [x[0] for x in text + image]
    all_metas = [x[1] for x in text + image]
    all_dists = [x[2] for x in text + image]

    deduped = deduplicate_results(all_docs, all_metas, all_dists)

    return {
        **state,
        "used_contexts": deduped
    }

def summarize_all(state: State) -> State:
    all_contexts = state.get("used_contexts", [])
    prompt = build_prompt("Give a summary of all information available.", all_contexts)
    result = llm.invoke(prompt).content
    return {
        **state,
        "answer": result
    }

def answer_with_context(state: State) -> State:
    prompt = build_prompt(state["query"], state.get("used_contexts", []))
    result = llm.invoke(prompt).content
    return {
        **state,
        "answer": result
    }

# ==== Build LangGraph ====

def build_graph():
    workflow = StateGraph(State)

    # Nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("retrieve", retrieve_contexts)
    workflow.add_node("summarize_all", summarize_all)
    workflow.add_node("answer_with_context", answer_with_context)

    # Flow
    workflow.set_entry_point("classify")
    workflow.add_conditional_edges("classify", lambda s: s["query_type"], {
        "summarize_all": "retrieve",
        "answer_with_context": "retrieve",
        "other": END
    })


    # Routing based on classifier
    workflow.add_conditional_edges("retrieve", lambda s: s["query_type"], {
        "summarize_all": "summarize_all",
        "answer_with_context": "answer_with_context",
        "other": END
    })

    workflow.add_edge("summarize_all", END)
    workflow.add_edge("answer_with_context", END)

    return workflow.compile()

# ==== Public function ====

graph = build_graph()

def run_agentic_rag(
    query: str,
    image: Optional[Union[str, Image.Image]] = None
) -> Dict:
    state = graph.invoke({
        "query": query,
        "image": image
    })

    if not state:
        return {
            "query": query,
            "answer": "Maaf, tidak ada jawaban yang bisa ditemukan.",
            "used_contexts": []
        }
    
    chat_id = create_new_chat(topic=query)
    if chat_id:
        save_chat_contents(
            chat_id=chat_id,
            query=query,
            answer=state["answer"],
            used_contexts=state.get("used_contexts", [])
        )

    return {
        "query": query,
        "answer": state.get("answer", "Maaf, tidak ada jawaban yang bisa ditemukan."),
        "used_contexts": state.get("used_contexts", [])
    }