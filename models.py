import os
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import wandb

class DoubleConv(nn.Module):
    """U-Net style double convolution block"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetEncoder(nn.Module):
    """Encoder part of U-Net architecture"""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        # Embedding layer for transfer learning
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        return x4, [x1, x2, x3, x4], self.embedding(x4)

class UNetDecoder(nn.Module):
    """Decoder part of U-Net architecture"""
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x, features):
        x = self.up1(x)
        x = torch.cat([x, features[2]], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, features[1]], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, features[0]], dim=1)
        x = self.conv3(x)
        
        return self.final(x)

class ButterflyAutoencoder(L.LightningModule):
    def __init__(self, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        encoded, features, _ = self.encoder(x)
        decoded = self.decoder(encoded, features)
        return decoded

    def _shared_step(self, batch, batch_idx, phase="train"):
        """Procesa un batch de datos"""
        x, _ = batch  # El Dataset ahora siempre devuelve una tupla
        x = x.to(self.device)
        
        # Añadir ruido y reconstruir
        noisy_x = x + torch.randn_like(x) * 0.1
        reconstructed = self(noisy_x)
        loss = F.mse_loss(reconstructed, x)
        
        # Logging
        self.log(f'{phase}_loss', loss, prog_bar=True, sync_dist=True)
        
        # Log imágenes periódicamente
        if batch_idx % 100 == 0 and phase == "train":
            try:
                wandb.log({
                    f"{phase}/original": wandb.Image(x[0].cpu()),
                    f"{phase}/noisy": wandb.Image(noisy_x[0].cpu()),
                    f"{phase}/reconstructed": wandb.Image(reconstructed[0].cpu())
                })
            except Exception as e:
                print(f"Warning: Could not log images: {e}")
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }

class ButterflyClassifier(L.LightningModule):
    """Classifier for butterfly species"""
    def __init__(self, 
                 num_classes: int = 20,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 pretrained_encoder: Optional[UNetEncoder] = None,
                 freeze_encoder: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_encoder'])
        
        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        else:
            self.encoder = UNetEncoder()
            
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        _, _, embeddings = self.encoder(x)
        return self.classifier(embeddings)

    def _calculate_metrics(self, preds, y):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y.cpu().numpy(),
            preds.cpu().numpy(),
            average='weighted',
            zero_division=0
        )
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def _shared_step(self, batch, batch_idx, phase="train"):
        x, y = batch
        # Asegurar que x es un tensor
        if not isinstance(x, torch.Tensor):
            x = torch.stack(x) if isinstance(x, list) else torch.tensor(x)
        x = x.to(self.device)
        
        # Asegurar que y es un tensor
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        y = y.to(self.device)
        
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        metrics = self._calculate_metrics(preds, y)
        self.log(f'{phase}_loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'{phase}_acc', acc, prog_bar=True, sync_dist=True)
        
        for metric_name, value in metrics.items():
            self.log(f'{phase}_{metric_name}', value, prog_bar=True, sync_dist=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        param_groups = [
            {'params': self.classifier.parameters(), 'lr': self.learning_rate}
        ]
        
        if self.encoder.requires_grad:
            param_groups.append(
                {'params': self.encoder.parameters(), 'lr': self.learning_rate * 0.1}
            )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

class ButterflyDataset(Dataset):
    """Dataset class for butterfly images"""
    def __init__(self, df: pd.DataFrame, transform, class_to_idx: dict, 
                 labeled: bool, data_dir: str):
        self.df = df
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.labeled = labeled
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['filepaths'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros((3, 224, 224))
        
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
            
        if self.labeled:
            label = torch.tensor(self.class_to_idx[row['labels']], dtype=torch.long)
            return image, label
        return image, image

class ButterflyDataModule(L.LightningDataModule):
    """DataModule for handling butterfly dataset"""
    def __init__(self,
                 data_dir: str = "Butterfly-dataset",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 labeled_ratio: float = 0.3):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.labeled_ratio = labeled_ratio
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def prepare_data(self):
        """Verify dataset existence"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

    def setup(self, stage: Optional[str] = None):
        """Setup train, validation and test datasets"""
        # Cargar y filtrar datos
        df = pd.read_csv(os.path.join(self.data_dir, "BUTTERFLIES.csv"))
        top_species = df['labels'].value_counts().nlargest(20).index
        df = df[df['labels'].isin(top_species)]
        
        # Crear mapeo de clases
        self.class_to_idx = {cls: idx for idx, cls in enumerate(top_species)}
        
        # Separar datos por conjunto
        train_df = df[df['data set'] == 'train']
        
        # Separar datos etiquetados y no etiquetados para cada especie
        labeled_dfs = []
        unlabeled_dfs = []
        
        for species in top_species:
            species_df = train_df[train_df['labels'] == species]
            n_labeled = int(len(species_df) * self.labeled_ratio)
            n_labeled = max(20, min(n_labeled, len(species_df) - 20))
            
            labeled_dfs.append(species_df.iloc[:n_labeled])
            unlabeled_dfs.append(species_df.iloc[n_labeled:])
        
        self.labeled_df = pd.concat(labeled_dfs)
        self.unlabeled_df = pd.concat(unlabeled_dfs)
        self.test_df = df[df['data set'] == 'test']
        self.val_df = df[df['data set'] == 'valid']

        # Mover muestras al conjunto de prueba
        for species in top_species:
            train_samples = self.labeled_df[self.labeled_df['labels'] == species]
            if len(train_samples) >= 20:
                samples_to_move = train_samples.iloc[:20].copy()
                samples_to_move['data set'] = 'test'
                self.labeled_df = self.labeled_df.drop(samples_to_move.index)
                self.test_df = pd.concat([self.test_df, samples_to_move])

    def _create_dataloader(self, df: pd.DataFrame, labeled: bool, shuffle: bool = True):
        dataset = ButterflyDataset(
            df=df,
            transform=self.transform,
            class_to_idx=self.class_to_idx,
            labeled=labeled,
            data_dir=self.data_dir
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )

    def train_dataloader(self, labeled: bool = True):
        """Return training dataloader"""
        df = self.labeled_df if labeled else self.unlabeled_df
        return self._create_dataloader(df, labeled=labeled)

    def val_dataloader(self):
        """Return validation dataloader"""
        return self._create_dataloader(self.val_df, labeled=True, shuffle=False)

    def test_dataloader(self):
        """Return test dataloader"""
        return self._create_dataloader(self.test_df, labeled=True, shuffle=False)

    def get_class_names(self) -> List[str]:
        """Return list of class names"""
        return list(self.class_to_idx.keys())

    def get_num_classes(self) -> int:
        """Return number of classes"""
        return len(self.class_to_idx)

    def get_sample_shape(self) -> Tuple[int, int, int]:
        """Return shape of input samples (channels, height, width)"""
        return (3, 224, 224)