# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import wandb
import torchvision
from typing import List, Tuple, Optional, Dict, Any
import torchmetrics

class UNetBlock(nn.Module):
    """Bloque básico de U-Net con skip connections."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetAutoencoder(nn.Module):
    """Autoencoder basado en arquitectura U-Net con skip connections."""
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        
        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_dim, 1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(latent_dim, 512, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(1024, 512)  # 1024 porque concatenamos con skip connection
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        
        return out, [e1, e2, e3, e4, b]

class ButterflyClassifier(nn.Module):
    """Clasificador CNN para mariposas."""
    def __init__(self, num_classes: int = 20):
        super().__init__()
        
        # Arquitectura simple entrenada desde cero
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

class LitAutoencoder(L.LightningModule):
    """Módulo Lightning para entrenar el autoencoder."""
    def __init__(self, autoencoder: nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.autoencoder = autoencoder
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['autoencoder'])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder(x)[0]  # Solo retornamos la reconstrucción
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x_hat, _ = self.autoencoder(x)
        loss = F.mse_loss(x_hat, x)
        
        # Log para wandb
        self.log('train_loss', loss)
        
        if batch_idx % 100 == 0:  # Log imágenes cada 100 batches
            self._log_reconstructions(x, x_hat)
        
        return loss
    
    def _log_reconstructions(self, x: torch.Tensor, x_hat: torch.Tensor):
        """Log de reconstrucciones para visualización."""
        # Selecciona algunas imágenes para visualizar
        idx = torch.randint(0, x.shape[0], (min(x.shape[0], 4),))
        x = x[idx]
        x_hat = x_hat[idx]
        
        # Desnormaliza las imágenes
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = x * std + mean
        x_hat = x_hat * std + mean
        
        # Log a wandb
        self.logger.experiment.log({
            "reconstructions": [
                wandb.Image(torch.cat([img, recon], dim=2))
                for img, recon in zip(x, x_hat)
            ]
        })
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }

class LitClassifier(L.LightningModule):
    """Módulo Lightning para entrenar los clasificadores."""
    def __init__(
        self,
        classifier: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.classifier = classifier
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        
        # Métricas
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=20)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=20)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=20)
        
        self.save_hyperparameters(ignore=['classifier', 'class_weights'])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, y, weight=self.class_weights.to(y.device))
        else:
            loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)
        
        # Log de métricas
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

# Crea un clasificador basado en el encoder del autoencoder
def create_encoder_classifier(autoencoder: UNetAutoencoder, num_classes: int, freeze_encoder: bool = True) -> nn.Module:
    """Crea un clasificador usando el encoder del autoencoder."""
    encoder_layers = [
        autoencoder.enc1,
        nn.MaxPool2d(2, 2),
        autoencoder.enc2,
        nn.MaxPool2d(2, 2),
        autoencoder.enc3,
        nn.MaxPool2d(2, 2),
        autoencoder.enc4,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    ]
    
    classifier = nn.Sequential(*encoder_layers)
    
    if freeze_encoder:
        for param in classifier[:-1].parameters():
            param.requires_grad = False
    
    return classifier