
import json
import csv
from datetime import datetime
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

# Configuración del modelo de evaluación (Ollama)
class CustomOllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3:8b"):
        self.model_name = "llama3:8b"
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
    """Carga el dataset desde un archivo JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_results_to_csv(results, dataset, output_filename, add_timestamp=False):
    """
    Guarda los resultados en un archivo CSV.
    
    Args:
        results: Resultados de la evaluación
        dataset: Dataset original
        output_filename: Nombre base del archivo (se añade .csv si no está presente)
        add_timestamp: Si True, añade marca de tiempo al nombre del archivo
    """
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
        #'contextual_recall_score',
        #'contextual_recall_reason',
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
                #'contextual_recall_score': results['contextual_recall'][i]['score'],
                #'contextual_recall_reason': results['contextual_recall'][i]['reason'],
                'contextual_precision_score': results['contextual_precision'][i]['score'],
                'contextual_precision_reason': results['contextual_precision'][i]['reason']
            }
            writer.writerow(row)
    
    print(f"\n✅ Resultados guardados en: {csv_filename}")
    return csv_filename

def evaluate_rag_dataset(dataset_path, output_filename="rag_results", model_name="mistral", add_timestamp=False):
    """
    Evalúa un dataset RAG y guarda los resultados.
    
    Args:
        dataset_path: Ruta al archivo JSON con los datos
        output_filename: Nombre base para el archivo de resultados
        model_name: Nombre del modelo Ollama a usar
        add_timestamp: Si True, añade timestamp al nombre del archivo
    """
    # Cargar datos
    try:
        dataset = load_dataset(dataset_path)
        print(f"📊 Dataset cargado con {len(dataset)} ejemplos")
    except Exception as e:
        print(f"❌ Error cargando el dataset: {str(e)}")
        return None
    
    # Inicializar modelo y métricas
    eval_model = CustomOllamaModel(model_name=model_name)
    
    metrics = {
        'faithfulness': FaithfulnessMetric(threshold=0.7, model=eval_model, include_reason=True),
        'answer_relevancy': AnswerRelevancyMetric(threshold=0.7, model=eval_model, include_reason=True),
        #'contextual_recall': ContextualRecallMetric(threshold=0.7, model=eval_model, include_reason=True),
        'contextual_precision': ContextualPrecisionMetric(threshold=0.7, model=eval_model, include_reason=True)
    }
    
    results = {metric: [] for metric in metrics}
    results['overall_scores'] = {}
    
    # Evaluar cada ejemplo
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
    
    # Calcular promedios
    for metric_name in metrics:
        scores = [r['score'] for r in results[metric_name]]
        results['overall_scores'][metric_name] = sum(scores) / len(scores)
    
    # Guardar resultados
    csv_file = save_results_to_csv(results, dataset, output_filename, add_timestamp)
    
    # Mostrar resumen
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluador de sistemas RAG')
    parser.add_argument('--dataset', required=True, help='Ruta al archivo JSON con el dataset')
    parser.add_argument('--output', default='rag_results', help='Nombre base para el archivo de salida (sin extensión)')
    parser.add_argument('--model', default='mistral', help='Modelo Ollama a usar para evaluación')
    parser.add_argument('--timestamp', action='store_true', help='Añadir timestamp al nombre del archivo')
    
    args = parser.parse_args()
    
    print("🚀 Iniciando evaluación RAG...")
    results = evaluate_rag_dataset(
        dataset_path=args.dataset,
        output_filename=args.output,
        model_name=args.model,
        add_timestamp=args.timestamp
    )
    
    if results:
        print(f"\n🎉 Evaluación completada! Resultados en: {results['csv_file']}")