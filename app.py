# app.py
import streamlit as st
import pdfplumber
import faiss
import numpy as np
import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="CAELUS SVA", layout="wide")
st.title("Sistema de Vivificação Autônoma (SVA)")
st.markdown("### Busca Semântica + Grafo Interativo")

DATA_DIR = Path("docs")
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 200
OVERLAP = 50

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)
        i += CHUNK_SIZE - OVERLAP
    return chunks

@st.cache_resource
def build_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return index, data["chunks"], data["sources"]

    model = load_model()
    all_chunks = []
    all_sources = []
    all_embs = []

    for pdf_file in DATA_DIR.glob("*.pdf"):
        text = extract_text(pdf_file)
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = model.encode(chunks)
        all_embs.append(embeddings)
        all_chunks.extend(chunks)
        all_sources.extend([pdf_file.name] * len(chunks))

    if not all_embs:
        # índice vazio
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        return index, [], []

    embeddings = np.vstack(all_embs).astype('float32')
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks, "sources": all_sources}, f)

    return index, all_chunks, all_sources

index, chunks, sources = build_index()

query = st.text_input("Faça uma pergunta:", placeholder="O que é Design Lúcido?")

if query:
    model = load_model()
    q_vec = model.encode([query]).astype('float32')
    faiss.normalize_L2(q_vec)
    if index.ntotal == 0:
        st.warning("Índice vazio — verifique se há PDFs em docs/ e se o index foi gerado.")
    else:
        D, I = index.search(q_vec, 10)
        results = []
        for d, i in zip(D[0], I[0]):
            if i < len(chunks):
                results.append((chunks[i], sources[i], d))

        for chunk, src, score in results:
            with st.expander(f"**{src}** – {score:.1%}"):
                st.write(chunk)

        if len(results) > 1:
            rel_idx = [chunks.index(r[0]) for r in results]
            rel_emb = model.encode([r[0] for r in results])
            sim = cosine_similarity(rel_emb)
            G = nx.Graph()
            for i, idx in enumerate(rel_idx):
                G.add_node(idx, title=results[i][0][:200], source=results[i][1])
            for i in range(len(rel_idx)):
                for j in range(i+1, len(rel_idx)):
                    if sim[i][j] > 0.6:
                        G.add_edge(rel_idx[i], rel_idx[j], weight=float(sim[i][j]))

            net = Network(height="500px", width="100%")
            net.from_nx(G)
            net.show("graph.html")
            with open("graph.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=550)