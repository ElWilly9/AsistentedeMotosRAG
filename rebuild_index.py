"""
Reconstruye la base vectorial Chroma con los embeddings E5 (prefijos query:/passage:).
Solo reindexa: no necesita modelo LLM ni API keys.
Uso:  python rebuild_index.py
"""
import logging
from asistente_rag import load_and_chunk_pdfs, create_or_load_vector_store

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

EMBEDDING = "intfloat/multilingual-e5-base"
PERSIST_DIR = "./chroma_db_intfloat"

if __name__ == "__main__":
    print("1/3  Cargando y troceando PDFs (con OCR en páginas de imagen)...")
    chunks = load_and_chunk_pdfs()
    print(f"     -> {len(chunks)} fragmentos generados.")

    print("2/3  Recreando base vectorial con embeddings E5...")
    create_or_load_vector_store(
        chunks,
        persist_directory=PERSIST_DIR,
        force_reload=True,
        embedding=EMBEDDING,
    )

    print("3/3  Listo. Base reconstruida en", PERSIST_DIR)
