# src/data.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
import pandas as pd
from torchvision import transforms
from PIL import Image
from typing import Optional, Dict, List
import numpy as np

class ButterflyDataset(Dataset):
    """Dataset para imágenes de mariposas."""
    def __init__(self, data_dir: str, df: pd.DataFrame, transform=None):
        self.data_dir = data_dir
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> tuple:
        img_path = os.path.join(self.data_dir, self.df.iloc[idx]['filepaths'])
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['class index']

        if self.transform:
            image = self.transform(image)

        return image, label

class ButterflyDataModule(L.LightningDataModule):
    """Módulo de datos Lightning para el dataset de mariposas."""
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        unlabeled_ratio: float = 0.7,
        image_size: int = 224,
        num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.unlabeled_ratio = unlabeled_ratio
        self.num_workers = num_workers
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def prepare_data(self):
        """Verifica que los datos existan."""
        csv_path = os.path.join(self.data_dir, 'BUTTERFLIES.csv')
        assert os.path.exists(csv_path), f"CSV file not found at {csv_path}"
        assert os.path.exists(self.data_dir), f"Data directory not found at {self.data_dir}"
    
    def setup(self, stage: Optional[str] = None):
        """Prepara los datasets para train/val/test."""
        # Lee el CSV
        csv_path = os.path.join(self.data_dir, 'BUTTERFLIES.csv')
        df = pd.read_csv(csv_path)
        
        # Selecciona top 20 clases
        class_counts = df['labels'].value_counts()
        self.top_20_classes = class_counts.head(20).index
        df = df[df['labels'].isin(self.top_20_classes)]
        
        # Mapea labels a índices
        unique_labels = sorted(df['labels'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        df['class index'] = df['labels'].map(self.label_to_idx)
        
        # Split train/val/test
        train_df = df[df['data set'] == 'train']
        val_df = df[df['data set'] == 'valid']
        test_df = df[df['data set'] == 'test']
        
        # Mueve 20 muestras de train a test para cada clase
        for label in self.top_20_classes:
            samples_to_move = train_df[train_df['labels'] == label].sample(20, random_state=42)
            train_df = train_df.drop(samples_to_move.index)
            test_df = pd.concat([test_df, samples_to_move])
        
        # Split datos de entrenamiento en labeled y unlabeled
        labeled_dfs = []
        unlabeled_dfs = []
        
        for label in self.top_20_classes:
            class_df = train_df[train_df['labels'] == label]
            unlabeled_count = int(len(class_df) * self.unlabeled_ratio)
            
            unlabeled = class_df.sample(unlabeled_count, random_state=42)
            labeled = class_df.drop(unlabeled.index)
            
            labeled_dfs.append(labeled)
            unlabeled_dfs.append(unlabeled)
        
        self.train_labeled_df = pd.concat(labeled_dfs)
        self.train_unlabeled_df = pd.concat(unlabeled_dfs)
        self.val_df = val_df
        self.test_df = test_df
        
        # Guarda estadísticas importantes
        self.num_classes = len(self.top_20_classes)
        self.class_weights = self._calculate_class_weights(self.train_labeled_df)
    
    def _calculate_class_weights(self, df: pd.DataFrame) -> torch.Tensor:
        """Calcula pesos de clase para manejar desbalance."""
        class_counts = df['class index'].value_counts().sort_index()
        total = len(df)
        weights = 1.0 / (class_counts / total)
        return torch.FloatTensor(weights)
    
    def train_dataloader(self) -> DataLoader:
        dataset = ButterflyDataset(self.data_dir, self.train_labeled_df, self.transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        dataset = ButterflyDataset(self.data_dir, self.val_df, self.transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        dataset = ButterflyDataset(self.data_dir, self.test_df, self.transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def unlabeled_dataloader(self) -> DataLoader:
        dataset = ButterflyDataset(self.data_dir, self.train_unlabeled_df, self.transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    @property
    def class_names(self) -> List[str]:
        """Retorna la lista de nombres de clases."""
        return list(self.label_to_idx.keys())