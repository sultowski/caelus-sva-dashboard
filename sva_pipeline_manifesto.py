
#!/usr/bin/env python3
"""
SVA Pipeline - Manifesto -> Semantic Vector Architecture (production-ready script)

Final file: sva_pipeline_manifesto.py
Purpose: Extract text from a PDF manifesto, generate embeddings (sentence-transformers),
build a semantic graph (NetworkX), persist vectors (FAISS optional), and export an
interactive HTML visualization (PyVis). Designed to run locally.

Requirements (install once):
    pip install -r requirements.txt

requirements.txt (recommended contents):
    sentence-transformers
    torch
    pdfplumber
    networkx
    scikit-learn
    pyvis
    faiss-cpu   # optional, use faiss-gpu if available
    python-frontmatter  # optional for metadata

Basic usage:
    python sva_pipeline_manifesto.py --pdf "/path/to/manifesto_origem.pdf" --outdir "./output"

Key features:
    - Robust PDF text extraction with fallback
    - Text cleaning and semantic chunking with overlap
    - Embeddings generation via SentenceTransformers
    - Similarity graph construction with cosine thresholding and edge weights
    - Optional FAISS index persistence
    - Interactive HTML visualization exported via PyVis
    - Logging and configurable hyperparameters

Author: Gerado automaticamente por um assistente (adaptar se necessário)
"""

import os
import sys
import argparse
import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional

# Lazy imports for heavy libraries (import only when needed)
def import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception as e:
        raise ImportError("Erro ao importar sentence-transformers. Instale via 'pip install sentence-transformers torch'") from e

def import_pdfplumber():
    try:
        import pdfplumber
        return pdfplumber
    except Exception as e:
        raise ImportError("Erro ao importar pdfplumber. Instale via 'pip install pdfplumber'") from e

def import_networkx():
    try:
        import networkx as nx
        return nx
    except Exception as e:
        raise ImportError("Erro ao importar networkx. Instale via 'pip install networkx'") from e

def import_sklearn():
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        return cosine_similarity, np
    except Exception as e:
        raise ImportError("Erro ao importar sklearn / numpy. Instale via 'pip install scikit-learn numpy'") from e

def import_pyvis():
    try:
        from pyvis.network import Network
        return Network
    except Exception as e:
        raise ImportError("Erro ao importar pyvis. Instale via 'pip install pyvis'") from e

def import_faiss():
    try:
        import faiss
        return faiss
    except Exception as e:
        return None  # FAISS optional

# -----------------------
# Utilities & Text Tools
# -----------------------
import re
def clean_text(text: str) -> str:
    """Basic cleaning: normalize spaces, remove control chars, fix hyphenation lines."""
    if not text:
        return ""
    # remove nulls and control chars
    text = text.replace('\x00', ' ')
    # fix hyphenation at line endings: "exam-\nple" -> "example"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # replace newlines with spaces, but keep paragraph boundaries (double newlines)
    text = text.replace('\r', '\n')
    text = re.sub(r'\n{2,}', '\n\n', text)
    # collapse remaining single newlines into space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def chunk_text_by_tokens(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Simple tokenizer-approx chunker based on whitespace tokens.
    chunk_size and overlap are in tokens (approx by split on whitespace).
    Returns list of chunks (strings).
    """
    if not text:
        return []
    tokens = text.split()
    chunks = []
    i = 0
    n = len(tokens)
    while i < n:
        j = min(i + chunk_size, n)
        chunk = " ".join(tokens[i:j])
        chunks.append(chunk)
        if j == n:
            break
        i = j - overlap
    return chunks

# -----------------------
# PDF Extraction
# -----------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    pdfplumber = import_pdfplumber()
    texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
    except Exception as e:
        logging.warning(f"Falha na extração com pdfplumber: {e}. Tentando fallback com leitura em binário simples.")
        try:
            # fallback: read as binary and attempt naive decode
            with open(pdf_path, "rb") as f:
                raw = f.read()
                texts.append(raw.decode("utf-8", errors="ignore"))
        except Exception as e2:
            logging.error(f"Fallback também falhou: {e2}")
            raise
    full_text = "\n\n".join(texts)
    return clean_text(full_text)

# -----------------------
# Embeddings & Indexing
# -----------------------
def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    SentenceTransformer = import_sentence_transformers()
    logging.info(f"Carregando modelo de embeddings: {model_name}")
    model = SentenceTransformer(model_name)
    return model

def embed_texts(model, texts: List[str], batch_size: int = 32):
    if not texts:
        return []
    # sentence-transformers returns numpy arrays
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    return embeddings

def build_faiss_index(embeddings, index_path: Optional[str] = None):
    faiss = import_faiss()
    import numpy as np
    if faiss is None:
        logging.warning("FAISS não disponível; pulando indexação com FAISS.")
        return None
    dim = embeddings.shape[1]
    # L2 index with normalization recommended for cosine similarity use-case
    index = faiss.IndexFlatIP(dim)
    # normalize vectors to unit norm for cosine-as-inner-product
    emb_np = embeddings.astype('float32')
    faiss.normalize_L2(emb_np)
    index.add(emb_np)
    if index_path:
        faiss.write_index(index, index_path)
        logging.info(f"FAISS index salvo em: {index_path}")
    return index

# -----------------------
# Graph construction
# -----------------------
def build_semantic_graph(text_chunks: List[str], embeddings, threshold: float = 0.65, top_k: int = None):
    nx = import_networkx()
    cosine_similarity, np = import_sklearn()
    n = len(text_chunks)
    G = nx.Graph()
    # add nodes with metadata
    for i, txt in enumerate(text_chunks):
        G.add_node(i, text=txt, title=(txt[:300] + '...') if len(txt) > 300 else txt)
    # compute pairwise similarities in batches if needed
    sims = cosine_similarity(embeddings)
    # optionally sparsify: only edges above threshold or top_k neighbors
    for i in range(n):
        if top_k:
            # get top_k indices excluding self
            row = sims[i]
            idx_sorted = np.argsort(-row)
            count = 0
            for j in idx_sorted:
                if i == j:
                    continue
                if count >= top_k:
                    break
                score = float(row[j])
                if score > 0:
                    G.add_edge(i, j, weight=score)
                    count += 1
        else:
            for j in range(i+1, n):
                score = float(sims[i][j])
                if score >= threshold:
                    G.add_edge(i, j, weight=score)
    return G

def export_graph_pyvis(G, outpath_html: str, notebook: bool = False, physics: bool = True, max_nodes: int = 1000):
    Network = import_pyvis()
    net = Network(height="900px", width="100%", notebook=notebook)
    net.barnes_hut()
    # add nodes
    for node, data in G.nodes(data=True):
        title = data.get("title", "")
        label = str(node)
        net.add_node(node, label=label, title=title)
    # add edges
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 0.0)
        net.add_edge(u, v, value=float(weight))
    net.toggle_physics(physics)
    # Reduce exported file size by limiting nodes if necessary
    if G.number_of_nodes() > max_nodes:
        logging.warning(f"Graph size ({G.number_of_nodes()}) > max_nodes ({max_nodes}). Export will include all nodes but could be heavy.")
    net.show(outpath_html)
    return outpath_html

# -----------------------
# Main pipeline
# -----------------------
def pipeline(pdf_path: str,
             outdir: str,
             model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
             chunk_size: int = 200,
             chunk_overlap: int = 50,
             similarity_threshold: float = 0.65,
             top_k: Optional[int] = None,
             faiss_index_path: Optional[str] = None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Pipeline iniciado. PDF: {pdf_path} | OutDir: {outdir}")

    # 1) Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise RuntimeError("Nenhum texto extraído do PDF. Verifique o arquivo.")
    logging.info(f"Texto extraído: {len(text)} caracteres")

    # 2) Split into paragraphs (preserve empty lines as paragraph separators)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    logging.info(f"Parágrafos detectados: {len(paragraphs)}")

    # 3) Further chunk paragraphs into semantic chunks using token-approx chunker
    chunks = []
    for p in paragraphs:
        if len(p.split()) <= chunk_size:
            chunks.append(p)
        else:
            chunks.extend(chunk_text_by_tokens(p, chunk_size=chunk_size, overlap=chunk_overlap))
    logging.info(f"Chunks finais: {len(chunks)}")

    # Save chunks for traceability
    chunks_path = outdir / "chunks.txt"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(f"## CHUNK {i}\n")
            f.write(c + "\n\n")
    logging.info(f"Chunks salvos em: {chunks_path}")

    # 4) Load model & generate embeddings
    model = load_embedding_model(model_name)
    embeddings = embed_texts(model, chunks)
    import numpy as np
    embeddings_np = np.array(embeddings, dtype='float32')
    logging.info(f"Embeddings gerados: shape={embeddings_np.shape}")

    # 5) Optionally persist FAISS index
    if faiss_index_path:
        idx = build_faiss_index(embeddings_np, index_path=faiss_index_path)
        if idx:
            logging.info("FAISS index criado com sucesso.")
    else:
        idx = None

    # 6) Build semantic graph
    G = build_semantic_graph(chunks, embeddings_np, threshold=similarity_threshold, top_k=top_k)
    logging.info(f"Grafo construído: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # 7) Export graph interactive HTML
    html_out = outdir / "sva_manifesto_graph.html"
    export_graph_pyvis(G, str(html_out))
    logging.info(f"Visualização HTML salva em: {html_out}")

    # 8) Persist metadata, embeddings and graph
    import json
    meta = {
        "pdf": os.path.basename(pdf_path),
        "model": model_name,
        "chunks": len(chunks),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    }
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    # Save embeddings as npy
    np.save(outdir / "embeddings.npy", embeddings_np)
    # Save graph using networkx gpickle
    nx = import_networkx()
    nx.write_gpickle(G, outdir / "graph.gpickle")
    logging.info("Artefatos persistidos: embeddings.npy, graph.gpickle, meta.json")

    return {
        "outdir": str(outdir),
        "html": str(html_out),
        "meta": meta
    }

# -----------------------
# CLI
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SVA pipeline: manifesto PDF -> semantic graph")
    parser.add_argument("--pdf", required=True, help="Caminho para o arquivo PDF do manifesto")
    parser.add_argument("--outdir", default="./output", help="Diretório de saída para artefatos")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Nome do modelo SentenceTransformers")
    parser.add_argument("--chunk_size", type=int, default=200, help="Tamanho do chunk (em tokens aproximados)")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Overlap entre chunks (tokens aproximados)")
    parser.add_argument("--similarity_threshold", type=float, default=0.65, help="Limiar de similaridade para criar arestas")
    parser.add_argument("--top_k", type=int, default=None, help="Se definido, cria no máximo top_k arestas por nó ao invés de usar threshold")
    parser.add_argument("--faiss_index", default=None, help="Se definido, salva índice FAISS em caminho fornecido")
    parser.add_argument("--loglevel", default="INFO", help="Logging level")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    try:
        result = pipeline(
            pdf_path=args.pdf,
            outdir=args.outdir,
            model_name=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            similarity_threshold=args.similarity_threshold,
            top_k=args.top_k,
            faiss_index_path=args.faiss_index
        )
        logging.info("Pipeline concluído com sucesso.")
        logging.info(f"Artefatos em: {result['outdir']}")
        logging.info(f"HTML: {result['html']}")
    except Exception as e:
        logging.exception("Pipeline falhou:")
        sys.exit(1)

if __name__ == "__main__":
    main()
