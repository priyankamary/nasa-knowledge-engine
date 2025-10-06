#!/usr/bin/env python3
import os
import uuid
from typing import Dict, Any, List

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import (
    HOST, PORT, CHROMA_DB_PATH, EMBED_MODEL, LLM_MODEL_ID,
    TOP_K, POOL_K
)

# ---------------- Utilities ----------------
def _fmt_snippets(docs, max_chars=400) -> str:
    rows = []
    for d in docs:
        title = d.metadata.get("Title") or d.metadata.get("title") or "Untitled"
        link  = d.metadata.get("Link")  or d.metadata.get("link")  or "N/A"
        text  = (d.page_content or "").replace("\n", " ")[:max_chars]
        rows.append(f"- {title} | {link}\n  {text}")
    return "\n".join(rows) if rows else "(no snippets)"

def _distinct_by_link(docs):
    seen = set(); out = []
    for d in docs:
        link = d.metadata.get("Link") or d.metadata.get("link")
        if not link or link in seen: 
            continue
        seen.add(link); out.append(d)
    return out

# ---------------- Prompts ----------------
SUMMARY_PROMPT = PromptTemplate.from_template("""
You are an expert NASA literature assistant.
Given the user's query and the retrieved snippets, write a crisp 4–6 sentence summary
covering the key ideas that directly answer or scope the question. If information is missing, say so.
Avoid speculation.

User Query:
{query}

Retrieved Snippets:
{snippets}

Concise summary:
""")

PAPER_CHAT_PROMPT = PromptTemplate.from_template("""
You are discussing ONE paper only. Use ONLY the provided paper snippets.
If the answer is not present, say so clearly and do not speculate.

Context so far:
- Original Query: {orig_query}
- System Summary: {system_summary}

Paper Title: {paper_title}
Paper Link:  {paper_link}

Paper Snippets:
{snippets}

User message:
{message}

Grounded answer:
""")

# ---------------- Load VectorStore ----------------
print(f"[serve] Loading embeddings: {EMBED_MODEL}")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

print(f"[serve] Attaching Chroma at: {CHROMA_DB_PATH}")
vectorstore = Chroma(
    collection_name="nasa_docs",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH,
)

# ---------------- Load LLM (TinyLlama 4-bit) ----------------
print(f"[serve] Loading LLM: {LLM_MODEL_ID} (4-bit NF4)")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gen_pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=1.0,
    max_new_tokens=256,
    repetition_penalty=1.05,
)
from langchain.llms import HuggingFacePipeline as _HFPipeline
llm = _HFPipeline(pipeline=gen_pipe)

# ---------------- Flask App ----------------
app = Flask(__name__)
CORS(app)

SESSIONS: Dict[str, Dict[str, Any]] = {}

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/search")
def search():
    body = request.get_json(force=True) or {}
    query = (body.get("query") or "").strip()
    k_top = int(body.get("k_top", TOP_K))
    pool_k = int(body.get("pool_k", POOL_K))
    if not query:
        return jsonify({"error":"Missing 'query'"}), 400

    # 1) pool for summary
    pool = vectorstore.similarity_search_with_score(query, k=pool_k)
    pool_docs = _distinct_by_link([d for d,_ in pool])
    snippets = _fmt_snippets(pool_docs)

    # 2) summary
    summary = LLMChain(llm=llm, prompt=SUMMARY_PROMPT).run(
        {"query": query, "snippets": snippets}
    ).strip()

    # 3) top-3 by distance (lower=better), distinct by link
    raw_top = vectorstore.similarity_search_with_score(query, k=max(k_top*3, k_top))
    picked, used = [], set()
    for d,dist in raw_top:
        title = d.metadata.get("Title") or d.metadata.get("title") or "Untitled"
        link  = d.metadata.get("Link")  or d.metadata.get("link")  or "N/A"
        if link not in used:
            picked.append({"title": title, "link": link, "distance": float(dist)})
            used.add(link)
        if len(picked) == k_top:
            break

    session_id = uuid.uuid4().hex[:8]
    SESSIONS[session_id] = {
        "orig_query": query,
        "system_summary": summary,
        "top3": picked,
        "history": []
    }
    return jsonify({"session_id": session_id, "summary": summary, "top3": picked})

@app.post("/chat")
def chat():
    body = request.get_json(force=True) or {}
    session_id = body.get("session_id", "")
    message = (body.get("message") or "").strip()
    paper_idx = body.get("paper_idx")
    paper_link = body.get("paper_link")
    k_chunks = int(body.get("k_chunks", 8))

    if session_id not in SESSIONS:
        return jsonify({"error":"Invalid session_id"}), 400
    if not message:
        return jsonify({"error":"Missing 'message'"}), 400

    sess = SESSIONS[session_id]
    top3 = sess.get("top3", [])

    if paper_link:
        sel = next((t for t in top3 if t["link"] == paper_link), None)
        if not sel: return jsonify({"error":"paper_link not in session top3"}), 400
    else:
        if not paper_idx or not (1 <= int(paper_idx) <= len(top3)):
            return jsonify({"error":"Provide valid 'paper_idx' or 'paper_link'"}), 400
        sel = top3[int(paper_idx)-1]

    link = sel["link"]; title = sel["title"]

    # Restrict to this paper’s chunks by Link metadata
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_chunks, "filter": {"Link": link}}
        )
        docs = retriever.get_relevant_documents(message)
    except Exception:
        pool = vectorstore.similarity_search_with_score(message, k=k_chunks*3)
        docs = [d for d,_ in pool if (d.metadata.get("Link") or d.metadata.get("link")) == link][:k_chunks]

    snippets = _fmt_snippets(docs, max_chars=450)
    answer = LLMChain(llm=llm, prompt=PAPER_CHAT_PROMPT).run({
        "orig_query": sess["orig_query"],
        "system_summary": sess["system_summary"],
        "paper_title": title,
        "paper_link": link,
        "snippets": snippets,
        "message": message
    }).strip()

    sess["history"].append(("user", message))
    sess["history"].append(("assistant", answer))

    grounding = [{
        "title": d.metadata.get("Title") or d.metadata.get("title") or "Untitled",
        "link":  d.metadata.get("Link")  or d.metadata.get("link")  or "N/A"
    } for d in docs]

    return jsonify({"answer": answer, "grounding": grounding})

@app.post("/related")
def related():
    body = request.get_json(force=True) or {}
    session_id = body.get("session_id", "")
    paper_idx = body.get("paper_idx")
    paper_link = body.get("paper_link")
    k_related = int(body.get("k_related", 5))
    seed_k = int(body.get("seed_k", 8))

    if session_id not in SESSIONS:
        return jsonify({"error":"Invalid session_id"}), 400
    sess = SESSIONS[session_id]; top3 = sess.get("top3", [])

    if paper_link:
        sel = next((t for t in top3 if t["link"] == paper_link), None)
        if not sel: return jsonify({"error":"paper_link not in session top3"}), 400
    else:
        if not paper_idx or not (1 <= int(paper_idx) <= len(top3)):
            return jsonify({"error":"Provide valid 'paper_idx' or 'paper_link'"}), 400
        sel = top3[int(paper_idx)-1]

    link = sel["link"]; title = sel["title"]

    # Build a seed from selected paper chunks
    try:
        base_docs = vectorstore.similarity_search(title, k=seed_k, filter={"Link": link})
    except Exception:
        pool = vectorstore.similarity_search_with_score(title, k=seed_k*2)
        base_docs = [d for d,_ in pool if (d.metadata.get("Link") or d.metadata.get("link")) == link][:seed_k]

    seed_text = " ".join([(d.page_content or "")[:400] for d in base_docs]) or title

    neighs = vectorstore.similarity_search_with_score(seed_text, k=k_related+5)
    related = []
    used = {link}
    for d,dist in neighs:
        lnk = d.metadata.get("Link") or d.metadata.get("link") or ""
        if lnk and lnk not in used:
            ttl = d.metadata.get("Title") or d.metadata.get("title") or "Untitled"
            related.append({"title": ttl, "link": lnk, "distance": float(dist)})
            used.add(lnk)
        if len(related) >= k_related:
            break

    return jsonify({"paper": {"title": title, "link": link}, "related": related})

if __name__ == "__main__":
    print(f"[serve] Starting on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
