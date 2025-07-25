# from langchain_community.llms import Ollama
# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from langchain.embeddings import HuggingFaceEmbeddings
# from ragas.evaluation import evaluate
# from ragas.metrics import faithfulness, answer_relevancy
# from datasets import Dataset
# import json
# import pandas as pd

# # ⚙ Instancia el modelo Ollama
# ollama_llm = Ollama(model="llama3:8b")  # Puedes usar otro como mistral, llama2, etc.

# # ⚙ Wrappers
# llm = LangchainLLMWrapper(ollama_llm)

# embed_model = LangchainEmbeddingsWrapper(
#     HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# )

# # ⚙ Asigna modelos a las métricas
# faithfulness.llm = llm
# faithfulness.embeddings = embed_model

# answer_relevancy.llm = llm
# answer_relevancy.embeddings = embed_model

# #  Leer el archivo generado por el sistema
# with open("ragas/ragas_history_E1L1.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 3. Convertir a DataFrame
# df = pd.DataFrame(data)

# # ⚠ Usa batch_size pequeño
# results = evaluate(
#     df,  # tu Dataset ya limpio con 'question', 'contexts', 'answer' y 'ground_truth'
#     metrics=[faithfulness, answer_relevancy],
#     batch_size=1
# )

# print(results)


import json
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from datasets import Dataset
import os


# Configuración de modelos
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = OllamaLLM(model="mistral", temperature=0.5)

# Cargar datos
with open("ragas/ragas_history_E2L3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Corregir formatos incorrectos
for item in data:
    # Si contexts es string, convertirlo a lista de un elemento
    if isinstance(item.get("contexts"), str):
        item["contexts"] = [item["contexts"]]
    # Asegurar que siempre sea lista
    item["contexts"] = item.get("contexts", [""]) or [""]  # Lista vacía -> [""]

# Crear Dataset
dataset = Dataset.from_dict({
    "question": [d["question"] for d in data],
    "answer": [d["answer"] for d in data],
    "contexts": [d["contexts"] for d in data],
    "ground_truth": [d["ground_truth"] for d in data]
})

# Evaluación
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=llm,
    embeddings=embed_model
)


print("\nResultados de la evaluación RAGAS:")
print(results)


score =results
df = score.to_pandas()
df.to_csv('ragas_resultsE2L3_m.csv', index=False)
