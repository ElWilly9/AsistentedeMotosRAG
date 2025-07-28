import json
import csv
from datetime import datetime
from deepeval_eval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

# CONFIGURACIÓN FIJA AQUÍ:
dataset_path = "rag_list/rag_list_history_E2L3.json"  # <-- Cambia esto por la ruta real del archivo a evaluar
output_filename = "resultados_deepeval/deepeval_E2L3l"  # <-- Carpeta y nombre base del archivo de salida CSV
model_name = "llama3:8b"  # <-- Cambia esto por el nombre del modelo Ollama que uses (ej: llama3, mistral)
add_timestamp = False  # <-- True para agregar timestamp al CSV, False si no lo quieres

# Clase para evaluación con Ollama
class CustomOllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
    
    def load_model(self):
        from ollama import Client
        return Client(host=self.base_url)
    
    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.generate(model=self.model_name, prompt=prompt)
        return response['response']
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        return self.model_name

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_results_to_csv(results, dataset, output_filename, add_timestamp=False):
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{output_filename}_{timestamp}.csv"
    else:
        csv_filename = output_filename if output_filename.endswith('.csv') else f"{output_filename}.csv"
    
    fieldnames = [
        'question',
        'faithfulness_score',
        'faithfulness_reason',
        'answer_relevancy_score',
        'answer_relevancy_reason',
        'contextual_precision_score',
        'contextual_precision_reason',
        'answer',
        'ground_truth',
        'contexts'
    ]
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, item in enumerate(dataset):
            row = {
                'question': item['question'],
                'answer': item['answer'],
                'ground_truth': item['ground_truth'],
                'contexts': str(item['contexts'])[:500] + '...' if len(str(item['contexts'])) > 500 else str(item['contexts']),
                'faithfulness_score': results['faithfulness'][i]['score'],
                'faithfulness_reason': results['faithfulness'][i]['reason'],
                'answer_relevancy_score': results['answer_relevancy'][i]['score'],
                'answer_relevancy_reason': results['answer_relevancy'][i]['reason'],
                'contextual_precision_score': results['contextual_precision'][i]['score'],
                'contextual_precision_reason': results['contextual_precision'][i]['reason']
            }
            writer.writerow(row)
    
    print(f"\n✅ Resultados guardados en: {csv_filename}")
    return csv_filename

def evaluate_rag_dataset(dataset_path, output_filename="rag_results", model_name="mistral", add_timestamp=False):
    try:
        dataset = load_dataset(dataset_path)
        print(f"📊 Dataset cargado con {len(dataset)} ejemplos")
    except Exception as e:
        print(f"❌ Error cargando el dataset: {str(e)}")
        return None
    
    eval_model = CustomOllamaModel(model_name=model_name)
    
    metrics = {
        'faithfulness': FaithfulnessMetric(threshold=0.7, model=eval_model, include_reason=True),
        'answer_relevancy': AnswerRelevancyMetric(threshold=0.7, model=eval_model, include_reason=True),
        'contextual_precision': ContextualPrecisionMetric(threshold=0.7, model=eval_model, include_reason=True)
    }
    
    results = {metric: [] for metric in metrics}
    results['overall_scores'] = {}
    
    print("\n🔍 Evaluando respuestas...")
    for i, item in enumerate(dataset, 1):
        print(f"  Procesando ejemplo {i}/{len(dataset)}...", end='\r')
        
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=item["answer"],
            expected_output=item["ground_truth"],
            retrieval_context=item["contexts"] if isinstance(item["contexts"], list) else [item["contexts"]]
        )
        
        for metric_name, metric in metrics.items():
            try:
                metric.measure(test_case)
                results[metric_name].append({
                    'score': metric.score,
                    'success': metric.success,
                    'reason': metric.reason
                })
            except Exception as e:
                print(f"\n⚠️ Error en {metric_name} para pregunta '{item['question'][:30]}...': {str(e)}")
                results[metric_name].append({
                    'score': 0.0,
                    'success': False,
                    'reason': f"Error: {str(e)}"
                })
    
    for metric_name in metrics:
        scores = [r['score'] for r in results[metric_name]]
        results['overall_scores'][metric_name] = sum(scores) / len(scores)
    
    csv_file = save_results_to_csv(results, dataset, output_filename, add_timestamp)
    
    print("\n📝 Resumen de evaluación:")
    print("=" * 50)
    for metric, score in results['overall_scores'].items():
        print(f"{metric.replace('_', ' ').title():<20}: {score:.3f}")
    print("=" * 50)
    
    return {
        'overall_scores': results['overall_scores'],
        'detailed_results': results,
        'csv_file': csv_file
    }

if __name__ == "__main__":
    print("🚀 Iniciando evaluación RAG...")
    results = evaluate_rag_dataset(
        dataset_path=dataset_path,
        output_filename=output_filename,
        model_name=model_name,
        add_timestamp=add_timestamp
    )
    
    if results:
        print(f"\n🎉 Evaluación completada! Resultados en: {results['csv_file']}")