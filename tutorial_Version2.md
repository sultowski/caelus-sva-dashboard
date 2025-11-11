```markdown
# PASSO A PASSO COMPLETO E SIMPLES
para criar, configurar e publicar o dashboard CAELUS SVA no GitHub + Streamlit Cloud — mesmo se você estiver com dificuldade.

---

## OBJETIVO FINAL
Você terá um link público tipo:  
`https://seu-nome-caelus-sva-dashboard.streamlit.app`  
com busca semântica + grafo interativo, rodando **grátis e online**!

---

## PASSO 1: Crie uma conta no GitHub (se ainda não tiver)

1. Acesse: https://github.com
2. Clique em **"Sign up"**
3. Preencha email, senha e nome de usuário (ex: `joao-caelus`)
4. Confirme o email

---

## PASSO 2: Crie o repositório no GitHub

1. No GitHub, clique no **+** (canto superior direito) → **"New repository"**
2. Preencha:

| Campo | Valor |
|------|-------|
| Repository name | `caelus-sva-dashboard` |
| Description | Dashboard CAELUS com busca semântica |
| Public | **Sim** (obrigatório para deploy grátis) |
| Initialize with README | **Marque** |

3. Clique em **"Create repository"**

---

## PASSO 3: Baixe e instale o Git (se ainda não tiver)

- Windows: https://git-scm.com/download/win
- Mac: já vem com o Xcode ou instale via Homebrew: `brew install git`
- Linux: `sudo apt install git`

---

## PASSO 4: Clone o repositório no seu computador

Abra o **Terminal** (ou Prompt de Comando) e digite:

```bash
git clone https://github.com/sultowski/caelus-sva-dashboard.git
cd caelus-sva-dashboard
```

> Substitua `sultowski` pelo seu nome no GitHub se você estiver usando outro usuário.

---

## PASSO 5: Crie os arquivos necessários

### 1. Crie a pasta `docs`

```bash
mkdir docs
```

### 2. Adicione seus PDFs

Coloque **dentro da pasta `docs`**:
- `manifesto_origem.pdf`
- `Modelo de Negócio CAELUS TechLab Rentável.pdf`

> Dica: Arraste os arquivos do seu computador para dentro da pasta `docs` no Explorador de Arquivos.

---

### 3. Crie o arquivo `app.py`

1. Abra o **Bloco de Notas** (ou VS Code)
2. Copie **todo o código presente em app.py** (fornecido separadamente)
3. Salve como `app.py` **dentro da pasta `caelus-sva-dashboard`**

(O arquivo `app.py` contém o código para carregar PDFs, criar embeddings com sentence-transformers, índice FAISS, busca semântica e grafo interativo via pyvis.)

---

### 4. Crie o arquivo `requirements.txt`

Salve na mesma pasta com as dependências necessárias (já fornecido).

---

### 5. Crie `.gitignore` (para não subir PDFs grandes)

Conteúdo sugerido (já fornecido).

---

## PASSO 6: Envie tudo para o GitHub

No terminal (dentro da pasta):

```bash
git add .
git commit -m "Dashboard CAELUS SVA pronto"
git push origin main
```

---

## PASSO 7: Deploy no Streamlit Cloud (GRÁTIS)

1. Acesse: https://share.streamlit.io
2. Faça login com **GitHub**
3. Clique em **"New app"**
4. Preencha:

| Campo | Valor |
|------|-------|
| Repository | `sultowski/caelus-sva-dashboard` |
| Branch | `main` |
| Main file path | `app.py` |

5. Clique em **"Deploy"**

Pronto! Em 2–5 minutos, seu app estará no ar!

---

## LINK FINAL (exemplo)

```
https://joao-caelus-caelus-sva-dashboard.streamlit.app
```

---

## TESTE RÁPIDO

Digite no campo:
- `O que é SVA?`
- `Gnose-as-a-Service`
- `Design Lúcido`

Você verá:
- Chunks relevantes
- Grafo interativo com conexões

---

## PRONTO! VOCÊ CONSEGUIU!

| Etapa | Status |
|------|--------|
| GitHub | Concluído |
| Código | Concluído |
| PDFs | Concluído |
| Deploy | Concluído |

---

## DÚVIDAS COMUNS

| Problema | Solução |
|--------|--------|
| "App não inicia" | Verifique se `app.py` está na raiz |
| "PDF não lido" | Verifique se está em `docs/` |
| "Erro de memória" | Use menos PDFs ou chunks menores |
| "Quero atualizar" | Edite, `git add .`, `git commit`, `git push` → redeploy automático |

---

## PRÓXIMOS PASSOS (opcional)

| Recurso | Como fazer |
|-------|-----------|
| Adicionar mais PDFs | Coloque em `docs/` e faça push |
| Mudar tema | Edite `st.set_page_config` |
| Colocar senha | Use `streamlit-authenticator` |
| Compartilhar link | Copie a URL do Streamlit |

---
```