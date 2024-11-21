import os
import torch
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import time
from PIL import Image
from pathlib import Path

def setup_directories(base_dir: str, experiment_name: str) -> Dict[str, str]:
    """
    Creates necessary directories for the experiment.
    
    Args:
        base_dir: Base directory for all outputs
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary with paths to created directories
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    dirs = {
        'root': exp_dir,
        'models': os.path.join(exp_dir, 'models'),
        'plots': os.path.join(exp_dir, 'plots'),
        'logs': os.path.join(exp_dir, 'logs'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs

def clean_gpu_memory():
    """Limpia la memoria de la GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def verify_dataset_structure(data_dir: str) -> bool:
    """
    Verifica que el dataset tenga la estructura correcta.
    
    Args:
        data_dir: Ruta al directorio del dataset
        
    Returns:
        bool: True si la estructura es correcta, False en caso contrario
    """
    required_files = ["BUTTERFLIES.csv"]
    required_dirs = ["train", "test", "valid"]
    
    try:
        # Verificar archivo CSV
        csv_path = Path(data_dir) / "BUTTERFLIES.csv"
        if not csv_path.exists():
            print(f"Error: No se encontró {csv_path}")
            return False
            
        # Verificar directorios
        for dir_name in required_dirs:
            dir_path = Path(data_dir) / dir_name
            if not dir_path.is_dir():
                print(f"Error: No se encontró el directorio {dir_path}")
                return False
                
        return True
    except Exception as e:
        print(f"Error verificando estructura del dataset: {e}")
        return False

def plot_confusion_matrix(y_true: List, 
                         y_pred: List, 
                         class_names: List[str],
                         save_path: Optional[str] = None,
                         title: str = 'Confusion Matrix'):
    """
    Genera y guarda la matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        class_names: Nombres de las clases
        save_path: Ruta para guardar la figura
        title: Título del gráfico
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        if wandb.run is not None:
            wandb.log({title: wandb.Image(save_path)})
    else:
        plt.show()

def measure_inference_performance(model: torch.nn.Module,
                                test_loader: torch.utils.data.DataLoader,
                                num_runs: int = 100,
                                warmup_runs: int = 10,
                                device: str = 'cuda') -> Dict:
    """
    Mide el rendimiento de inferencia del modelo.
    
    Args:
        model: Modelo a evaluar
        test_loader: DataLoader de prueba
        num_runs: Número de ejecuciones para medir
        warmup_runs: Número de ejecuciones de calentamiento
        device: Dispositivo a utilizar
        
    Returns:
        Dict con métricas de rendimiento
    """
    model = model.to(device)
    model.eval()
    
    # Obtener un batch de ejemplo
    batch = next(iter(test_loader))
    if isinstance(batch, tuple):
        x, _ = batch
    else:
        x = batch
    x = x.to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
    
    # Medir rendimiento
    latencies = []
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc="Measuring inference time"):
            start_time = time.perf_counter()
            _ = model(x)
            latencies.append(time.perf_counter() - start_time)
    
    # Calcular métricas
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    throughput = 1.0 / avg_latency
    
    # Calcular tamaño del modelo
    model_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_size_mb = model_size / (1024 * 1024)
    
    return {
        "avg_latency": avg_latency,
        "std_latency": std_latency,
        "throughput": throughput,
        "model_size_mb": model_size_mb,
    }

def plot_training_curves(metrics: Dict[str, List], 
                        save_path: Optional[str] = None,
                        title_prefix: str = ""):
    """
    Genera gráficos de las curvas de entrenamiento.
    
    Args:
        metrics: Diccionario con métricas de entrenamiento
        save_path: Ruta para guardar la figura
        title_prefix: Prefijo para el título
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4*num_metrics))
    if num_metrics == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_title(f"{title_prefix} {metric_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        if wandb.run is not None:
            wandb.log({f"{title_prefix} Training Curves": wandb.Image(save_path)})
    else:
        plt.show()

def compare_model_variants(results: List[Dict],
                         metrics: List[str],
                         save_dir: str):
    """
    Compara y visualiza resultados entre variantes del modelo.
    
    Args:
        results: Lista de diccionarios con resultados por variante
        metrics: Lista de métricas a comparar
        save_dir: Directorio para guardar las visualizaciones
    """
    df_results = pd.DataFrame(results)
    
    # Crear gráfico de barras comparativo
    plt.figure(figsize=(15, 5*len(metrics)))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(len(metrics), 1, i)
        sns.barplot(data=df_results, x='variant', y=metric)
        plt.title(f'Comparison of {metric} across Model Variants')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    if wandb.run is not None:
        wandb.log({"Model Variants Comparison": wandb.Image(save_path)})
    
    # Guardar resultados en CSV
    df_results.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)

def save_experiment_config(config: Dict, save_dir: str):
    """
    Guarda la configuración del experimento.
    
    Args:
        config: Diccionario con la configuración
        save_dir: Directorio donde guardar la configuración
    """
    import json
    
    config_path = os.path.join(save_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    if wandb.run is not None:
        wandb.save(config_path)

def log_batch_examples(batch: torch.Tensor,
                      reconstructed: torch.Tensor,
                      original_title: str = "Original",
                      reconstructed_title: str = "Reconstructed"):
    """
    Registra ejemplos de imágenes originales y reconstruidas en wandb.
    
    Args:
        batch: Tensor con imágenes originales
        reconstructed: Tensor con imágenes reconstruidas
        original_title: Título para imágenes originales
        reconstructed_title: Título para imágenes reconstruidas
    """
    if wandb.run is None:
        return
        
    def tensor_to_images(tensor):
        images = []
        for img in tensor[:4]:  # Limitamos a 4 imágenes
            img = img.cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            images.append(wandb.Image(img))
        return images
    
    wandb.log({
        original_title: tensor_to_images(batch),
        reconstructed_title: tensor_to_images(reconstructed)
    })

def count_parameters(model: torch.nn.Module) -> int:
    """
    Cuenta el número total de parámetros entrenables en el modelo.
    
    Args:
        model: Modelo PyTorch
        
    Returns:
        Número total de parámetros entrenables
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_model_complexity(model: torch.nn.Module, 
                           input_size: tuple = (1, 3, 224, 224)) -> Dict:
    """
    Calcula la complejidad del modelo (FLOPs, parámetros, etc.).
    
    Args:
        model: Modelo PyTorch
        input_size: Tamaño de entrada para el modelo
        
    Returns:
        Dict con métricas de complejidad
    """
    from torchvision.models.feature_extraction import get_graph_node_names
    from torchvision.models._utils import _get_model_complexity_info
    
    def prepare_input(input_size):
        return torch.ones(()).new_empty(input_size)
    
    model.eval()
    input = prepare_input(input_size)
    
    flops = 0
    params = 0
    
    try:
        flops, params = _get_model_complexity_info(
            model, input_size[1:], as_strings=False,
            print_per_layer_stat=False, verbose=False
        )
    except Exception as e:
        print(f"Warning: Could not compute FLOPs: {e}")
        params = count_parameters(model)
    
    return {
        "flops": flops,
        "parameters": params,
        "parameters_millions": params / 1e6,
        "flops_billions": flops / 1e9
    }