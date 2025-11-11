# CAELUS SVA — Piloto (MVP) — Pacote de Deploy

Gerado: 2025-11-11T00:17:53.392130Z

Este pacote contém o código e os documentos necessários para rodar o **piloto CAELUS SVA** (Sistema de Vivificação Autônoma) localmente e para deploy no **Streamlit Cloud**.

### Arquivos principais incluídos
- `app_Version2.py` — Dashboard Streamlit (busca semântica + grafo interativo).
- `sva_pipeline_manifesto.py` — Pipeline CLI para extrair, chunkar e gerar grafo semântico a partir de um PDF.
- `requirements_Version2.txt` — Dependências para o dashboard.
- `requirements_sva_pipeline.txt` — Dependências para o pipeline (alternativa).
- `tutorial_Version2.md` — Guia passo-a-passo para deploy no GitHub + Streamlit Cloud.
- `gitignore_Version2.txt` — Sugestão de .gitignore (não subir PDFs, índices).
- `manifesto origem.pdf` — Documento fonte do manifesto (indexar com o pipeline).
- `Modelo de Negócio CAELUS TechLab.docx` — Documento de apoio.
- `grok_report.pdf` — Relatório adicional.

> Se algum arquivo importante estiver faltando verifique se foi nomeado corretamente ou movido.

---

## Passo rápido para deploy no Streamlit Cloud

1. Crie um repositório no GitHub e suba todo o conteúdo deste pacote na **branch main**.
2. No Streamlit Cloud (https://share.streamlit.io) clique em "New app".
3. Conecte com o repositório GitHub, branch `main` e informe `app_Version2.py` como main file.
4. Garanta que `requirements_Version2.txt` esteja na raiz do repo.
5. Deploy — o Streamlit cuidará do resto.

**Observações**
- Coloque os PDFs que deseja indexar na pasta `docs/` antes do deploy (ou adapte `app_Version2.py` para apontar para outra pasta).
- Para usar FAISS em ambientes com GPU ou limitações, considere ajustar `faiss-cpu` / `faiss-gpu`.
- Se quiser executar o pipeline localmente:
  ```
  python sva_pipeline_manifesto.py --pdf "manifesto origem.pdf" --outdir "./output"
  ```
- Se quiser uma versão limpa do app com nomes simples, renomeie `app_Version2.py` para `app.py` (opcional).

---

## Suporte rápido
Se desejar, eu posso:
- Converter `Modelo de Negócio CAELUS TechLab.docx` para PDF e incluir aqui.
- Renomear `app_Version2.py` → `app.py` automaticamente.
- Gerar o repositório GitHub e proceder com o deploy (necessita suas credenciais).

Boa sorte — o protótipo fica pronto com poucos cliques.
