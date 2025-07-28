import json
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from datasets import Dataset
from ragas.run_config import RunConfig
import os

# Configuración de modelos
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)
llm = OllamaLLM(model="mistral", temperature=0.5, timeout=720)

# Cargar datos
with open("rag_list/rag_list_history_E2L3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Crear Dataset
dataset = Dataset.from_dict({
    "question": [d["question"] for d in data],
    "answer": [d["answer"] for d in data],
    "contexts": [d["contexts"] for d in data],
    "ground_truth": [d["ground_truth"] for d in data]
})

run_config = RunConfig(timeout=60)
# Evaluación
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    run_config=run_config,
    batch_size=2,
    llm=llm,
    embeddings=embed_model
)

# Resultados
print("\nResultados de la evaluación RAGAS:")
print(results)

# Guardar CSV
df = results.to_pandas()
df.to_csv('resultados_ragas/ragas_resultsE2L3_m.csv', index=False)
