# 🏍️ Asistente de MotosRAG

Asistente Virtual Interactivo para resolución de inquietudes acerca de tu moto implementando RAG, creado e implementado por **William Andrés Velásquez Ruiz**, **Ángel Andrés Martínez Oñate** y **Juan Pablo Hoyos Sanchez**.

Este repositorio contiene el desarrollo de un **asistente virtual interactivo** para la recomendación de repuestos y la resolución de dudas técnicas sobre motocicletas, específicamente la **Bajaj Boxer CT100 KS**, utilizando un sistema **RAG (Retrieval-Augmented Generation)** integrado con modelos de lenguaje de gran escala (**LLMs**).

---

## 🚀 Descripción del Proyecto

Aprovechando los avances recientes en inteligencia artificial, se propone una solución local y personalizada que:

- Responde preguntas técnicas en lenguaje natural
- Recupera información precisa desde documentos oficiales (PDFs)
- Genera respuestas contextualizadas y confiables
- Evalúa automáticamente la calidad de las respuestas usando métricas estandarizadas

Todo esto se presenta mediante un **avatar 2D animado** que responde en texto y voz directamente en el navegador, ofreciendo una experiencia amigable y autónoma.

---

## 🧩 Tecnologías y Herramientas

- 🐍 **Python** (backend)
  - `LangChain` para orquestación del pipeline RAG
  - `pdfplumber` para extracción de texto y tablas
  - `Chroma` como base de datos vectorial persistente
  - `Flask` como servidor web
- 🌐 **HTML, CSS y JavaScript** (frontend)
  - Interfaz simple y funcional con entrada escrita
  - Reproducción automática de audio con reproductor HTML
- 🤖 **Modelos Usados**
  - **Embeddings**: `intfloat/multilingual-e5-base`, `paraphrase-multilingual-MiniLM-L12-v2`
  - **Modelos de lenguaje**: `Gemini-2.0-Flash`, `Llama 3.3-70b`, `Gemma2-9b-it`
- 📊 **Evaluación de Calidad**:
  - `RAGAS`: métricas como *faithfulness*, *context precision*, *answer correctness*
  - `DeepEval`: evaluación complementaria

---

## 📊 Resultados Obtenidos

Se evaluaron **6 combinaciones** de modelos de embeddings y generativos en tareas de recuperación y generación de información técnica. Algunos resultados destacados:

- ✅ **Mayor fidelidad factual**: `E1 + L2` (`intfloat/multilingual-e5-base` + `Llama 3.3-70b`)
- ✅ **Mayor precisión contextual**: `E2 + L1` (`MiniLM` + `Gemini`)
- ✅ **Respuestas más relevantes**: `E1 + L2` bajo evaluación con Mistral

> Se identificó que las herramientas automáticas de evaluación pueden fallar en preguntas extensas, por lo que se realizaron análisis filtrados para obtener métricas más representativas.

📌 **Configuración final del asistente**:  
El sistema quedó configurado con la combinación **E1L2**, al ofrecer un equilibrio óptimo entre fidelidad, precisión y relevancia.  
---

## 🛠 Instalación y Ejecución

> ⚠️ Asegúrate de tener instalado `Python 3.10.0` en tu entorno y tu archivo `.env` configurado correctamente con tu clave de acceso para Groq API. Puedes conseguirla en: [https://groq.com](https://groq.com)

```bash
git clone https://github.com/ElWilly9/AsistentedeMotosRAG.git
cd AsistentedeMotosRAG
pip install -r requirements.txt
python app.py
```

---

## 📖 Cómo citar este trabajo

Si este repositorio o el artículo asociado te fue útil en tu investigación, desarrollo o trabajo académico, puedes citarlo de la siguiente manera:

### Formato BibTeX

```bibtex
@article{velasquez2026asistente,
  author    = {William Andrés Velasquez Ruiz and Ángel Andrés Martínez Oñate and Juan Pablo Hoyos Sánchez},
  title     = {Asistente virtual interactivo para resolver consultas relacionadas con motocicletas mediante RAG},
  journal   = {Prospectiva},
  volume    = {24},
  number    = {1},
  year      = {2026},
  doi       = {10.15665/rp.v24i1.3828}
}
```

### IEEE

W. A. Velasquez Ruiz, Á. A. Martínez Oñate y J. P. Hoyos Sánchez,
“Asistente virtual interactivo para resolver consultas relacionadas con motocicletas mediante RAG,”
Prospectiva, vol. 24, no. 1, 2026, doi: 10.15665/rp.v24i1.3828.

### DOI
DOI: https://doi.org/10.15665/rp.v24i1.3828
