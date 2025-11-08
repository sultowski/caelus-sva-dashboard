# CAELUS SVA Dashboard

Dashboard semântico para o Sistema de Vivificação Autônoma (SVA) da CAELUS TechLab. Processa PDFs de manifestos e modelos de negócio, gera embeddings com SentenceTransformers, busca com FAISS e visualiza grafos com PyVis.

## Funcionalidades
- **Busca Semântica**: Encontre chunks relevantes em documentos CAELUS.
- **Grafo Interativo**: Visualização de similaridades entre resultados.
- **Suporte a Múltiplos PDFs**: Adicione arquivos em `/docs/`.

## Setup Local
1. Clone o repo: `git clone https://github.com/SEU_USERNAME/caelus-sva-dashboard.git`
2. Instale dependências: `pip install -r requirements.txt`
3. Adicione PDFs em `/docs/` (ex.: manifesto_origem.pdf).
4. Rode: `streamlit run app.py`

## Deploy no Streamlit Cloud
1. Vá para [share.streamlit.io](https://share.streamlit.io) e crie uma conta (conecte ao GitHub).
2. Clique "New app" > Selecione este repo > Branch `main` > Main file `app.py`.
3. Deploy automático! URL: `https://seu-username-caelus-sva-dashboard.streamlit.app`

## Estrutura
- `app.py`: App principal.
- `requirements.txt`: Dependências.
- `docs/`: Pasta para PDFs (gitignore'd para privacidade).
- `.github/workflows/ci.yml`: Testes automáticos.

## Contribuições
Fork e PR bem-vindos! Para issues, use o GitHub.

© 2025 CAELUS TechLab
