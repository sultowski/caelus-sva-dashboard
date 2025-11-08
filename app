# app.py
import streamlit as st
import os
import json
import numpy as np
import faiss
import pdfplumber
import networkx as nx
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.metrics.pairwise import cosine_similarity

# ====================== CONFIGURAÇÕES ======================
st.set_page_config(page_title="CAELUS SVA – Busca Semântica", layout="wide")
st.title("Sistema de Vivificação Autônoma (SVA)")
st.markdown("### Busca Semântica + Grafo de Conhecimento")

DATA_DIR = Path("docs")
INDEX_FILE = Path("faiss_index.bin")
CHUNKS_FILE = Path("chunks.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 10

# ====================== FUNÇÕES ======================
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_data
def extract_text_from_pdf(pdf_path: Path) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Erro ao ler {pdf_path.name}: {e}")
        return ""
    return text.strip()

def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    if not text.strip():
        return []
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        if i >= len(words) and i > 0:
            break
    return chunks

@st.cache_resource
def build_index():
    if INDEX_FILE.exists() and CHUNKS_FILE.exists():
        st.info("Carregando índice FAISS e chunks salvos...")
        index = faiss.read_index(str(INDEX_FILE))
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return index, data["chunks"], data["sources"]

    st.info("Construindo índice FAISS a partir dos PDFs...")
    model = load_model()
    all_chunks = []
    all_sources = []
    embeddings_list = []

    pdf_files = [f for f in DATA_DIR.glob("*.pdf") if not f.name.startswith("~$")]
    if not pdf_files:
        st.error("Nenhum PDF encontrado em `docs/`. Adicione seus arquivos.")
        st.stop()

    progress_bar = st.progress(0)
    for idx, pdf_file in enumerate(pdf_files):
        text = extract_text_from_pdf(pdf_file)
        if not text:
            continue
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue

        embeddings = model.encode(chunks, batch_size=16, show_progress_bar=False)
        embeddings_list.append(embeddings)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_sources.append(pdf_file.name)

        progress_bar.progress((idx + 1) / len(pdf_files))

    if not embeddings_list:
        st.error("Nenhum chunk gerado. Verifique os PDFs.")
        st.stop()

    embeddings_all = np.vstack(embeddings_list).astype('float32')
    faiss.normalize_L2(embeddings_all)

    dim = embeddings_all.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_all)

    # Salvar
    faiss.write_index(index, str(INDEX_FILE))
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks, "sources": all_sources}, f, ensure_ascii=False, indent=2)

    st.success(f"Índice FAISS criado: {len(all_chunks)} chunks de {len(pdf_files)} PDFs")
    return index, all_chunks, all_sources

def search(query: str, index, chunks, sources, k=TOP_K_RESULTS):
    model = load_model()
    query_vec = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= len(chunks):
            continue
        results.append({
            "chunk": chunks[idx],
            "source": sources[idx],
            "score": float(score)
        })
    return results

def build_graph(results, chunks, sources, threshold=0.65):
    if len(results) < 2:
        return None

    model = load_model()
    relevant_indices = []
    relevant_texts = []
    for r in results:
        try:
            idx = chunks.index(r["chunk"])
            relevant_indices.append(idx)
            relevant_texts.append(r["chunk"])
        except ValueError:
            continue

    if len(relevant_indices) < 2:
        return None

    embeddings = model.encode(relevant_texts)
    sim_matrix = cosine_similarity(embeddings)

    G = nx.Graph()
    for i, idx in enumerate(relevant_indices):
        G.add_node(idx, label=f"Chunk {idx}", title=relevant_texts[i][:300] + "...", source=sources[idx])

    for i in range(len(relevant_indices)):
        for j in range(i + 1, len(relevant_indices)):
            sim = sim_matrix[i][j]
            if sim >= threshold:
                G.add_edge(relevant_indices[i], relevant_indices[j], weight=sim)

    return G

def render_pyvis_graph(G):
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    net.force_atlas_2based()

    color_map = {
        "manifesto_origem.pdf": "#1f77b4",
        "Modelo de Negócio CAELUS TechLab Rentável.pdf": "#d62728"
    }

    for node in G.nodes():
        data = G.nodes[node]
        color = color_map.get(data["source"], "#7f7f7f")
        size = 20 if "SVA" in data["title"] else 15
        net.add_node(node, label=data["label"], title=data["title"], color=color, size=size)

    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, value=data["weight"] * 15, title=f"Sim: {data['weight']:.3f}", color="#888888")

    net.show("graph.html", notebook=False)
    return "graph.html"

# ====================== CARREGAR ÍNDICE ======================
try:
    index, chunks, sources = build_index()
except Exception as e:
    st.error(f"Erro ao carregar índice: {e}")
    st.stop()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Configurações")
    top_k = st.slider("Resultados", 3, 20, TOP_K_RESULTS)
    threshold = st.slider("Threshold Grafo", 0.5, 0.9, 0.65, 0.05)
    st.markdown("---")
    st.caption(f"Chunks indexados: {len(chunks)}")
    st.caption(f"Documentos: {len(set(sources))}")
    if st.button("Reprocessar Índice"):
        INDEX_FILE.unlink(missing_ok=True)
        CHUNKS_FILE.unlink(missing_ok=True)
        st.experimental_rerun()

# ====================== BUSCA ======================
query = st.text_input("Digite sua pergunta:", placeholder="Ex: O que é Design Lúcido?")

if query:
    with st.spinner("Buscando..."):
        results = search(query, index, chunks, sources, k=top_k)

    st.subheader(f"Resultados para: *{query}*")

    if not results:
        st.warning("Nenhum resultado encontrado.")
    else:
        for i, r in enumerate(results):
            score_pct = r['score'] * 100
            with st.expander(f"**{r['source']}** – Score: {score_pct:.1f}%"):
                st.write(r["chunk"])

    # ====================== GRAFO ======================
    if len(results) > 1:
        with st.spinner("Construindo grafo..."):
            G = build_graph(results, chunks, sources, threshold)
            if G and len(G.nodes) > 1:
                html_file = render_pyvis_graph(G)
                with open(html_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=650, scrolling=True)
            else:
                st.info("Grafo não gerado: poucos nós conectados ou similaridade baixa.")
    else:
        st.info("Poucos resultados para gerar grafo.")

# ====================== RODAPÉ ======================
st.markdown("---")
st.caption("CAELUS TechLab © 2025 | Gnose-as-a-Service (GaaS) | Powered by FAISS + SentenceTransformers")
