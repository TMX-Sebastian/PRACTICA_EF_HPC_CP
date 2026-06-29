import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from pymongo import MongoClient
from pymongo.collection import Collection
import cohere
import google.generativeai as genai

load_dotenv()

# =============================================================
# Configuración principal
# En Azure Web App for Container, estos valores se configuran en:
# Settings > Environment variables / Application settings.
# No subas llaves reales a GitHub.
# =============================================================
APP_USER = os.getenv("APP_USER", "Minchola Sebastian")
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://sebasm_db_user:Moxx6666@cluster-sebasm.izy2wyg.mongodb.net/")
MONGODB_DB = os.getenv("MONGODB_DB", "hpc_pdf_analytics")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "pdf_chunks")
MONGO_VECTOR_INDEX = os.getenv("MONGO_VECTOR_INDEX", "vector_index")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "jB2tJGmUBK9hhQ83T83r3ZUgLitaGrEQBeFeJKlK")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6Ix_CKpdWEuIY9S3-ayAIBoKtZ5oDRWcqfqsQPnVKSLqw")
COHERE_MODEL = os.getenv("COHERE_MODEL", "embed-multilingual-v3.0")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

st.set_page_config(
    page_title="PDF Semantic Search | HPC Cloud Native",
    page_icon="📄",
    layout="wide",
)


def missing_config() -> List[str]:
    missing = []
    for key, value in {
        "MONGO_URI": MONGO_URI,
        "COHERE_API_KEY": COHERE_API_KEY,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "APP_USER": APP_USER,
    }.items():
        if not value or value == "Apellido Nombre":
            missing.append(key)
    return missing


@st.cache_resource(show_spinner=False)
def get_collection() -> Collection:
    if not MONGO_URI:
        raise RuntimeError("Falta configurar MONGO_URI")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    collection = client[MONGODB_DB][MONGODB_COLLECTION]
    collection.create_index([("doc_id", 1), ("chunk_id", 1)], unique=True)
    collection.create_index([("uploaded_by", 1), ("source_file", 1)])
    return collection


@st.cache_resource(show_spinner=False)
def get_cohere_client() -> cohere.Client:
    if not COHERE_API_KEY:
        raise RuntimeError("Falta configurar COHERE_API_KEY")
    return cohere.Client(COHERE_API_KEY)


@st.cache_resource(show_spinner=False)
def configure_gemini() -> bool:
    if not GEMINI_API_KEY:
        raise RuntimeError("Falta configurar GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    return True


def extract_text_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Extrae texto por página desde un archivo PDF."""
    temp_path = "/tmp/uploaded_document.pdf"
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)

    reader = PdfReader(temp_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned_text = " ".join(text.split())
        if cleaned_text:
            pages.append({"page": page_number, "text": cleaned_text})
    return pages


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Divide texto largo en chunks con solapamiento."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks


def build_chunks(pages: List[Dict[str, Any]], source_file: str, doc_id: str) -> List[Dict[str, Any]]:
    records = []
    chunk_counter = 0
    for page in pages:
        for chunk in chunk_text(page["text"]):
            chunk_counter += 1
            records.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_counter,
                    "source_file": source_file,
                    "page": page["page"],
                    "text": chunk,
                    "uploaded_by": APP_USER,
                    "created_at": datetime.utcnow(),
                }
            )
    return records


def embed_texts(texts: List[str], input_type: str) -> List[List[float]]:
    """Genera embeddings usando Cohere."""
    client = get_cohere_client()
    response = client.embed(
        texts=texts,
        model=COHERE_MODEL,
        input_type=input_type,
    )
    return [list(map(float, emb)) for emb in response.embeddings]


def save_chunks_with_embeddings(collection: Collection, chunks: List[Dict[str, Any]]) -> int:
    if not chunks:
        return 0

    batch_size = 32
    saved = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [item["text"] for item in batch]
        embeddings = embed_texts(texts, input_type="search_document")
        for item, embedding in zip(batch, embeddings):
            item["embedding"] = embedding
            collection.replace_one(
                {"doc_id": item["doc_id"], "chunk_id": item["chunk_id"]},
                item,
                upsert=True,
            )
            saved += 1
    return saved


def cosine_similarity(a: List[float], b: List[float]) -> float:
    av = np.array(a, dtype=np.float32)
    bv = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom == 0:
        return 0.0
    return float(np.dot(av, bv) / denom)


def vector_search(collection: Collection, question: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], str]:
    """Busca contexto relevante en MongoDB Atlas.

    Primero intenta $vectorSearch. Si el índice vectorial todavía no existe,
    usa un fallback con similitud coseno en Python para que la app siga funcionando.
    """
    query_embedding = embed_texts([question], input_type="search_query")[0]

    pipeline = [
        {
            "$vectorSearch": {
                "index": MONGO_VECTOR_INDEX,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "source_file": 1,
                "page": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        results = list(collection.aggregate(pipeline))
        return results, "MongoDB Atlas Vector Search"
    except Exception:
        docs = list(
            collection.find(
                {"uploaded_by": APP_USER},
                {"_id": 0, "source_file": 1, "page": 1, "text": 1, "embedding": 1},
            ).limit(1000)
        )
        for doc in docs:
            doc["score"] = cosine_similarity(query_embedding, doc.get("embedding", []))
        docs = sorted(docs, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        return docs, "Fallback local con similitud coseno"


def generate_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    configure_gemini()
    model = genai.GenerativeModel(GEMINI_MODEL)

    context_text = "\n\n".join(
        [
            f"Fuente: {ctx.get('source_file')} | Página: {ctx.get('page')}\n{ctx.get('text')}"
            for ctx in contexts
        ]
    )

    prompt = f"""
Eres un asistente académico para una plataforma cloud-native de búsqueda semántica en PDFs.
Responde únicamente con base en el contexto recuperado. Si el contexto no alcanza, dilo de forma clara.

Contexto recuperado:
{context_text}

Pregunta del usuario:
{question}

Respuesta:
"""
    response = model.generate_content(prompt)
    return getattr(response, "text", "No se pudo generar una respuesta.")


# =============================================================
# Interfaz Streamlit
# =============================================================
st.title("📄 Plataforma Cloud-Native de Búsqueda Semántica en PDFs")
st.caption("Streamlit + Docker + Azure Web App for Container + MongoDB Atlas + Cohere + Gemini")

with st.sidebar:
    st.header("Configuración")
    st.write(f"**Usuario:** {APP_USER}")
    st.write(f"**Base de datos:** {MONGODB_DB}")
    st.write(f"**Colección:** {MONGODB_COLLECTION}")
    st.write(f"**Índice vectorial:** {MONGO_VECTOR_INDEX}")

    missing = missing_config()
    if missing:
        st.warning("Faltan variables: " + ", ".join(missing))
    else:
        st.success("Variables principales configuradas")

    st.divider()
    st.markdown("**Cambio visual CI/CD:** v1.0 - Examen Final HPC")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Ingesta y vectorización")
    uploaded_file = st.file_uploader("Sube un documento PDF", type=["pdf"])

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        doc_id = hashlib.sha256(pdf_bytes).hexdigest()
        st.info(f"Archivo listo: {uploaded_file.name}")

        if st.button("Procesar PDF", type="primary"):
            try:
                with st.spinner("Extrayendo texto, creando chunks y generando embeddings..."):
                    collection = get_collection()
                    pages = extract_text_from_pdf(pdf_bytes)
                    chunks = build_chunks(pages, uploaded_file.name, doc_id)
                    saved = save_chunks_with_embeddings(collection, chunks)
                st.success(f"PDF procesado correctamente. Chunks guardados: {saved}")
            except Exception as exc:
                st.error(f"No se pudo procesar el PDF: {exc}")

with col2:
    st.subheader("2. Chatbot con contexto")
    question = st.text_area(
        "Realiza una pregunta sobre el PDF procesado",
        placeholder="Ejemplo: ¿Cuál es el objetivo principal del documento?",
        height=120,
    )

    if st.button("Consultar chatbot"):
        if not question.strip():
            st.warning("Ingresa una pregunta antes de consultar.")
        else:
            try:
                with st.spinner("Buscando contexto y generando respuesta..."):
                    collection = get_collection()
                    contexts, method = vector_search(collection, question, top_k=5)
                    answer = generate_answer(question, contexts)
                st.success(f"Método de recuperación: {method}")
                st.markdown("### Respuesta")
                st.write(answer)

                with st.expander("Ver contexto recuperado"):
                    for i, ctx in enumerate(contexts, start=1):
                        st.markdown(f"**Contexto {i} | Página {ctx.get('page')} | Score {ctx.get('score', 0):.4f}**")
                        st.write(ctx.get("text"))
            except Exception as exc:
                st.error(f"No se pudo consultar el chatbot: {exc}")
