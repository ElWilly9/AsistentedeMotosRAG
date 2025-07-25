import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
import rank_bm25
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pdfplumber
from langchain.schema import Document
import pyttsx3
import logging
from voz_text import query_voz
from prueba import obtener_texto_por_voz


# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
api_key_groq = os.getenv("GROQ_API_KEY")  

# Asegúrate de tener .env con GEMINI_API_KEY

# Directorio con PDFs y archivo de historial
DOCS_DIR = "./data/"
HISTORY_FILE = "./chat_history.json"
PERSIST_DIR = "./chroma_db"

# Text Processing
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 100  

def limpiar_consola():
    os.system('cls' if os.name == 'nt' else 'clear')

# Paso 1: Cargar y chunkear PDFs
def load_and_chunk_pdfs():
    documents = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DOCS_DIR, filename)
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extraer texto
                    text = page.extract_text()
                    
                    # Extraer tablas
                    tables = page.extract_tables()
                    table_text = ""
                    for table in tables:
                        for row in table:
                            # Filtrar valores None y convertir a string
                            row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
                            table_text += row_text + "\n"
                    
                    # Combinar texto y tablas
                    combined_text = f"{text}\n\nTablas:\n{table_text}"
                    
                    # Crear documento con metadatos
                    doc = Document(
                        page_content=combined_text,
                        metadata={
                            "source": filename,
                            "page": page_num + 1
                        }
                    )
                    documents.append(doc)
    
    # Dividir documentos en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Paso 2: Crear o cargar base de datos vectorial
def create_or_load_vector_store(chunks, persist_directory=PERSIST_DIR, force_reload=False, embedding=""):
    if force_reload and os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)

    if os.path.exists(persist_directory) and os.listdir(persist_directory) and not force_reload:
        vector_store = Chroma(
            collection_name="bajaj_boxer",
            embedding_function=HuggingFaceEmbeddings(model_name=embedding),
            persist_directory=persist_directory
        )
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=HuggingFaceEmbeddings(model_name=embedding),
            collection_name="bajaj_boxer",
            persist_directory=persist_directory
        )
    return vector_store

'''def inicializar_Bm25(chunks):
    texts = [chunk.page_content for chunk in chunks]
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k = 5
    return bm25_retriever

def combine_serch(retrievers=[], weights=[]):
    hybrid_combine = EnsembleRetriever(retrievers=retrievers, weights=weights)
    return hybrid_combine
'''
# Paso 3: Cargar o inicializar historial
def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    return []

def save_chat_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[-5:], f, ensure_ascii=False, indent=2)  # Limitar a las últimas 10 interacciones

def save_ragas_history(question, answer, contexts, filename="ragas/ragas_history_E2L2.json"):
    """
    Guarda la pregunta, respuesta generada, contexto recuperado y un campo ground_truth vacío en un archivo JSON para evaluación con RAGAS.
    """
    entry = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ""
    }
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Paso 4: Configurar el sistema RAG con memoria
def setup_rag(vector_store, model, chunks):
    if model =="1":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0
        )
    elif model == "2":
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key_groq,
            temperature=0
        )
    elif model == "3":
        llm = ChatGroq(
            model="gemma2-9b-it",
            api_key=api_key_groq,
            temperature=0
        )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Prompt personalizado para tono natural y cordial
    prompt_template = """
    Eres un asistente experto en motocicletas Bajaj Boxer CT100 KS. 
    Tu trabajo es ayudar a los usuarios con información técnica, mantenimiento y operación de esta motocicleta específica.

    HISTORIAL DE LA CONVERSACIÓN:
    {chat_history}

    CONTEXTO RELEVANTE DE LA DOCUMENTACIÓN:
    {context}

    PREGUNTA DEL USUARIO: {question}

    INSTRUCCIONES PARA TU RESPUESTA:
    - no generes respuestas tan rapidas, se conciso 
    - Responde ÚNICAMENTE sobre la Bajaj Boxer CT100 KS y si hablaras de otra moto, especificalo
    - Usa un tono cordial, natural y profesional en español
    - Si el usuario te pregunta sobre otra moto, especificalo
    - No te extiendas en tu respuesta, a no ser que el usuario te lo pida y osea necesario
    - Basa tu respuesta en la información del contexto proporcionado y el historial de la conversación
    - Si la información está en una tabla, incluye los valores específicos
    - Si la información no está en el contexto, dilo claramente
    - Proporciona respuestas prácticas y útiles para el usuario
    - Incluye detalles técnicos relevantes cuando sea apropiado
    - Si mencionas especificaciones técnicas, cita los valores exactos
    - Mantén la respuesta completa pero concisa, no la hagas tan extensa

    RESPUESTA:
    """ 
    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=prompt_template
    )
    '''bm25 = inicializar_Bm25(chunks)
    inicializar_chroma = vector_store.as_retriever(search_kwargs={"k": 5})
    retriever = combine_serch(retrievers=[bm25, inicializar_chroma], weights=[0.5, 0.5])'''
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return rag_chain

def print_document_samples(chunks):
    print("\nMuestras de documentos cargados:")
    for i, chunk in enumerate(chunks[:3]):  # Mostrar primeros 3 chunks
        print(f"\nChunk {i+1}:")
        print(f"Fuente: {chunk.metadata.get('source')}")
        print(f"Página: {chunk.metadata.get('page')}")
        print("Contenido:")
        print(chunk.page_content[:200] + "...")  # Primeros 200 caracteres

def decir_respuesta(texto):
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.setProperty('volume', 1)
    os.makedirs("audio", exist_ok=True)
    engine.say(texto)
    engine.save_to_file(texto, "audio/answer_ia_voz.mp3")
    engine.runAndWait()

def responder_desde_rag(pregunta, vector_store, rag_chain):
    result = rag_chain.invoke({"question": pregunta})
    respuesta = result["answer"]

    contexts = "\n\n".join([doc.page_content for doc in result.get("source_documents", [])])
    save_ragas_history(pregunta, respuesta, contexts)
    decir_respuesta(respuesta)  # genera el MP3

    return respuesta


# Paso 5: Ciclo interactivo en consola
def main():
    print("¿Quieres (C)argar la base vectorial existente o (R)ecrearla con los documentos actuales?")
    op = input("Escribe C para cargar o R para recrear: ").strip().lower()
    # op2 = input("Que tipo de embeddings quieres usar? (1) intfloat/multilingual-e5-base (2) sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2:\n ")
    force_reload = (op == "r")

    # if op2 == "1":
    #     embeddings = "intfloat/multilingual-e5-base"
    # elif op2 == "2":
    #     embeddings = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # else:
    #     print("Opción no válida")
    embeddings = "intfloat/multilingual-e5-base"
    
    # model = input("Que modelo de IA quieres usar? (1) Gemini-2.0-flash (2) Llama-3.3-70b-versatile: (3) gemma2-9b-it: \n")
    model = "1"

    if force_reload:
        print("Cargando y procesando documentos...")
        chunks = load_and_chunk_pdfs()
        print(f"Se cargaron {len(chunks)} fragmentos de documentos.")
        print_document_samples(chunks)
    else:
        chunks = []

    if embeddings == "intfloat/multilingual-e5-base":
        PERSIST_DIR = "./chroma_db_intfloat"
    elif embeddings == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
        PERSIST_DIR = "./chroma_db_paraphrase"

    print("Cargando base de datos vectorial...")
    vector_store = create_or_load_vector_store(chunks, PERSIST_DIR, force_reload=force_reload, embedding=embeddings)
    print("Configurando sistema RAG con memoria...")

    rag_chain = setup_rag(vector_store, model, chunks)

    # Cargar historial previo
    chat_history = load_chat_history()
    if chat_history:
        print("\nHistorial de conversación cargado:")

    print("\n¡Asistente virtual para Bajaj Boxer CT100 KS listo!")
    print("Di tu pregunta (o 'salir' para terminar):")

    while True:
        query = query_voz()
        if not query:
            continue
        if query.lower() == "salir":
            break

        result = rag_chain.invoke({"question": query})
        respuesta = result["answer"]
        print("\nRespuesta 🤖:", respuesta)
        decir_respuesta(respuesta)

        # Guardar en historial
        chat_history.append({"question": query, "answer": respuesta})
        save_chat_history(chat_history)

        # Guardar para evaluación con RAGAS
        # El contexto recuperado está en result["source_documents"]
        contexts = "\n\n".join([doc.page_content for doc in result.get("source_documents", [])])
        save_ragas_history(query, respuesta, contexts)

        # Preguntar si desea hacer otra consulta o salir
        opcion = input("\n¿Quieres hacer otra consulta? (S para salir, cualquier otra tecla para hacer una nueva pregunta): ").strip().lower()
        if opcion == "s":
            break
        limpiar_consola()

    print("¡Gracias por usar el asistente!")
    limpiar_consola()

if __name__ == "__main__":
    logging.getLogger("pdfminer").setLevel(logging.ERROR)  # Para que no salgan los warnings de pdfplumber
    main()
    