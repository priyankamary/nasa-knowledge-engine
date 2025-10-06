#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd

from config import CHROMA_DB_PATH, DATA_CSV, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP, EMBED_MODEL

def main(csv_path: str, persist_dir: str):
    csv_path = Path(csv_path)
    assert csv_path.exists(), f"CSV not found: {csv_path}"
    print(f"Reading: {csv_path}")

    df = pd.read_csv(csv_path)
    # Normalize columns
    needed = {"title", "link", "abstract"}
    missing = needed - set(map(str.lower, df.columns))
    if missing:
        raise ValueError(f"CSV must have columns: title, link, abstract (missing: {missing})")
    # Ensure exact names
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df[["title", "link", "abstract"]]

    # Make column lookup case-insensitive
    lower_cols = {c.lower(): c for c in df.columns}
    def col(name): return lower_cols.get(name, name)  # returns actual column name if present
    
    required = ["title", "link", "abstract"]
    missing = [c for c in required if c not in lower_cols]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. Found columns: {list(df.columns)}")
    
    documents = []
    for _, row in df.iterrows():
        content = str(row[col("abstract")]).strip()
        meta = {
            "Title": str(row[col("title")]).strip(),
            "Link": str(row[col("link")]).strip(),
        }
        documents.append(Document(page_content=content, metadata=meta))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={TEXT_CHUNK_SIZE}, overlap={TEXT_CHUNK_OVERLAP}).")

    print(f"Initializing embeddings: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    print("Building Chroma index…")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="nasa_docs",
    )
    vs.persist()
    print(f"✅ Chroma ready at: {persist_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DATA_CSV, help="Path to papers.csv with title,link,abstract")
    parser.add_argument("--db", default=CHROMA_DB_PATH, help="Chroma persist directory")
    args = parser.parse_args()
    main(args.csv, args.db)
