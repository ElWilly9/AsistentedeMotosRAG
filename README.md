# Asistente de MotosRAG
Asistente Virtual Interactivo para resolución de inquietudes acerca de tu moto implementando RAG creado e implementado por William Andres Velasquez Ruiz y Angel Andres Martinez Oñate, como proyecto para la asignatira de inteligencia artificial.


Este repositorio contiene el desarrollo de un **asistente virtual interactivo** para la recomendación de repuestos y la resolución de dudas técnicas sobre motocicletas, específicamente la **Bajaj Boxer CT100 KS**, utilizando un sistema **RAG (Retrieval-Augmented Generation)** integrado con modelos de lenguaje de gran escala (**LLMs**).

## 🚀 Descripción del Proyecto

Aprovechando los avances recientes en inteligencia artificial, se propone una solución local y personalizada que:

- Responde preguntas técnicas en lenguaje natural
- Recupera información técnica precisa desde documentos oficiales (PDFs)
- Genera respuestas contextualizadas y confiables
- Evalúa automáticamente la calidad de las respuestas usando métricas estandarizadas

Todo esto se presenta mediante un **avatar 2D interactivo** que responde por voz directamente en el navegador.

## 🧩 Tecnologías y Herramientas

- 🐍 **Python** (backend)
  - `LangChain`
  - `pdfplumber` para extracción de texto y tablas
  - `Chroma` como vector store
  - `speech_recognition` para síntesis de voz
- 🌐 **HTML, CSS y JavaScript** (frontend)
  - Integración de avatar 2D animado en navegador
  - Reproducción de audio con `SpeechSynthesis`
- 🤖 **Modelos Usados**
  - Embeddings: `intfloat/multilingual-e5-base`, `paraphrase-multilingual-MiniLM-L12-v2`
  - LLMs: `Gemini-2.0-Flash`, `Llama 3.3-70b`, `Gemma2-9b-it`
- 📊 Evaluación:
  - `RAGAS` (faithfulness, context precision, answer correctness)
  - `DeepEval`

## 📊 Resultados Obtenidos

Se evaluaron **6 combinaciones** de embeddings y modelos de lenguaje en tareas de recuperación y generación. Algunos resultados destacados:

- 📌 **Mayor fidelidad factual:** `E1 + L2 (Llama 3.3-70b)`
- 📌 **Mejor precision de contexto:** `E2 + L1 (Gemini)`
- 📌 **Mayor relevancia de respuestas:** `E1 + L2` bajo evaluación con Mistral

> Se identificó que la evaluación automática puede fallar en preguntas extensas, por lo que se realizaron análisis filtrados para obtener métricas más representativas.

## 🛠 Instalación y Ejecución
Nota: Recuerda que para que funcione correctamente debes crear un archivo '''.env''' en el cual ingreses tu api key de groq, puedes conseguirla en el siguiente enlace: https://groq.com/
```bash
git clone https://github.com/ElWilly9/AsistentedeMotosRAG.git
cd tu_carpeta
pip install -r requirements.txt
python app.py


