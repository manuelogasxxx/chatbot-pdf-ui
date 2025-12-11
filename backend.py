from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

import os
import tempfile

app = FastAPI()

# ───────────────────────────────────────────────
# CORS (permite conectar HTML/JS desde localhost)
# ───────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────
# MODELO LLM
# ───────────────────────────────────────────────
llm = OllamaLLM(model="phi3")

# Embeddings
embed_model = FastEmbedEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB en RAM
vectorstore = None


# ───────────────────────────────────────────────
# API: Cargar y vectorizar un PDF
# ───────────────────────────────────────────────
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile):
    global vectorstore

    # Guardar archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        filepath = tmp.name
        tmp.write(await file.read())

    # Leer PDF
    loader = PyMuPDFLoader(filepath)
    data_pdf = loader.load()

    # Dividir en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500
    )

    docs = splitter.split_documents(data_pdf)

    # Crear VectorDB FAISS
    vectorstore = FAISS.from_documents(docs, embed_model)

    return {"status": "PDF procesado correctamente", "chunks": len(docs)}


# ───────────────────────────────────────────────
# Modelo para recibir preguntas
# ───────────────────────────────────────────────
class Question(BaseModel):
    query: str


# ───────────────────────────────────────────────
# API: Preguntar al sistema
# ───────────────────────────────────────────────
@app.post("/ask")
async def ask_question(q: Question):
    global vectorstore

    if vectorstore is None:
        return {"error": "Debes cargar un PDF primero"}

    # Buscar chunks relevantes
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_chunks = retriever.invoke(q.query)

    # Unir el contexto para el LLM
    context = "\n\n".join([c.page_content for c in relevant_chunks])

    prompt = f"""
Eres un asistente experto en análisis de documentos PDF.

Responde la siguiente pregunta SOLO usando el contexto:

--- CONTEXTO ---
{context}
-----------------

Pregunta: {q.query}

Respuesta:
"""

    answer = llm.invoke(prompt)

    return {
        "answer": answer,
        "chunks": [c.page_content for c in relevant_chunks]
    }
