from flask import Flask, request, jsonify
from flask_cors import CORS
from asistente_rag import (
    load_and_chunk_pdfs, create_or_load_vector_store, setup_rag,
    load_chat_history, save_chat_history, save_ragas_history
)

app = Flask(__name__)
CORS(app)

# Configuración
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
IA_MODEL = "1"  # Gemini Flash
FORCE_RELOAD = False

# Cargar documentos
chunks = load_and_chunk_pdfs() if FORCE_RELOAD else []

# Base vectorial
PERSIST_DIR = "./chroma_db_intfloat"
vector_store = create_or_load_vector_store(chunks, PERSIST_DIR, force_reload=FORCE_RELOAD, embedding=EMBEDDING_MODEL)

# Configurar RAG
rag_chain = setup_rag(vector_store, IA_MODEL, chunks)

# Historial
chat_history = load_chat_history()

@app.route("/preguntar", methods=["POST"])
def preguntar():
    data = request.get_json()
    pregunta = data.get("pregunta", "")

    if not pregunta:
        return jsonify({"error": "Pregunta vacía"}), 400

    result = rag_chain.invoke({"question": pregunta})
    respuesta = result["answer"]

    # Guardar historial
    chat_history.append({"question": pregunta, "answer": respuesta})
    save_chat_history(chat_history)

    # Guardar contexto para RAGAS
    contextos = "\n\n".join([doc.page_content for doc in result.get("source_documents", [])])
    save_ragas_history(pregunta, respuesta, contextos)

    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    app.run(debug=True)

