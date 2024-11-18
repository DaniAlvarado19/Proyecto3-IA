# src/utils.py
import torch
import time
import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import wandb
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix

def set_seed(seed: int = 42) -> None:
    """Establece semillas para reproducibilidad."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate_model_performance(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evalúa el rendimiento del modelo."""
    model.eval()
    model = model.to(device)
    
    # Inicializa métricas
    accuracy = Accuracy(task="multiclass", num_classes=20).to(device)
    precision = Precision(task="multiclass", num_classes=20).to(device)
    recall = Recall(task="multiclass", num_classes=20).to(device)
    f1 = F1Score(task="multiclass", num_classes=20).to(device)
    
    # Mide tiempo de inferencia
    latencies = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch, targets in dataloader:
            batch, targets = batch.to(device), targets.to(device)
            
            # Mide tiempo de inferencia
            start_time = time.time()
            outputs = model(batch)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
            
            preds = outputs.argmax(dim=1)
            all_preds.append(preds)
            all_targets.append(targets)
    
    # Concatena predicciones y targets
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Calcula métricas
    results = {
        'accuracy': accuracy(all_preds, all_targets).item(),
        'precision': precision(all_preds, all_targets).item(),
        'recall': recall(all_preds, all_targets).item(),
        'f1_score': f1(all_preds, all_targets).item(),
        'avg_latency': np.mean(latencies),
        'std_latency': np.std(latencies)
    }
    
    return results

def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """Calcula estadísticas del tamaño del modelo."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'parameter_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_size_mb': size_all_mb
    }

def plot_confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    class_names: List[str],
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """Genera una matriz de confusión."""
    cm = confusion_matrix(targets.cpu(), predictions.cpu())
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()

def compare_model_versions(
    original_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """Compara versiones original y cuantizada del modelo."""
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    original_perf = evaluate_model_performance(original_model, test_dataloader, device)
    quantized_perf = evaluate_model_performance(quantized_model, test_dataloader, device)
    
    results = {
        'size_reduction_ratio': 1 - (quantized_size['total_size_mb'] / original_size['total_size_mb']),
        'size_reduction_mb': original_size['total_size_mb'] - quantized_size['total_size_mb'],
        'accuracy_difference': quantized_perf['accuracy'] - original_perf['accuracy'],
        'latency_improvement': 1 - (quantized_perf['avg_latency'] / original_perf['avg_latency']),
        'original_metrics': original_perf,
        'quantized_metrics': quantized_perf,
        'original_size': original_size,
        'quantized_size': quantized_size
    }
    
    return results

def save_experiment_results(
    results: Dict[str, Any],
    save_dir: Path,
    experiment_name: str
) -> None:
    """Guarda los resultados del experimento."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Guarda resultados en formato JSON
    import json
    results_path = save_dir / f"{experiment_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Guarda visualizaciones
    figs_dir = save_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    
    # Guarda gráficos relevantes
    for name, fig in results.get('figures', {}).items():
        fig.savefig(figs_dir / f"{experiment_name}_{name}.png")
        plt.close(fig)

def log_batch_predictions(
    wandb_logger: wandb.sdk.wandb_run.Run,
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    class_names: List[str],
    step: int,
    max_images: int = 8
) -> None:
    """Log de predicciones para visualización en W&B."""
    # Selecciona un subconjunto de imágenes
    n_images = min(max_images, images.shape[0])
    indices = torch.randperm(images.shape[0])[:n_images]
    
    # Prepara las imágenes y etiquetas
    images = images[indices]
    true_labels = true_labels[indices]
    pred_labels = pred_labels[indices]
    
    # Desnormaliza las imágenes
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = images * std + mean
    
    # Crea captions para las imágenes
    captions = [
        f'True: {class_names[t]}\nPred: {class_names[p]}'
        for t, p in zip(true_labels, pred_labels)
    ]
    
    # Log a W&B
    wandb_logger.log({
        "predictions": [
            wandb.Image(img, caption=caption)
            for img, caption in zip(images, captions)
        ]
    }, step=step)