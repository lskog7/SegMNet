import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from segmentation_models_pytorch.losses import (
    BINARY_MODE,
    DiceLoss,
    MULTICLASS_MODE,
    SoftBCEWithLogitsLoss,
    SoftCrossEntropyLoss,
)
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, PolynomialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchvision import tv_tensors
from torchvision.transforms import v2

@dataclass
class Params:
    version: str = "0.1.0"

    # Конфиг самого обучения
    epochs: int = 50
    batch_size: int = 64
    num_workers: int = os.cpu_count()
    drop_last: bool = False
    learning_rate: float = 0.02
    grad_clip_value: float = 1.0
    early_stopping: bool = False
    early_stopping_patience: int = 0
    lr_warmup_epochs: int = 0
    scheduler_patience: int = 0
    lr_min: float = 1e-08
    min_epochs: int = 10

    # Конфиг модели
    pretrained: bool = True
    precision: torch.dtype = "bf16-mixed"
    amp: bool = True
    encoder_name: str = "efficientnet-b0"
    encoder_depth: int = 5
    encoder_weights: Optional[str] = "imagenet"
    decoder_channels: List[int] = (256, 128, 64, 32, 16)
    decoder_use_batchnorm: bool = True  # True, False, "inplace"
    decoder_attention_type: str | None = None  # None or "scse"
    in_channels: int = 1
    num_classes: int = 4
    activation: Optional[str] = None
    aux_params: Optional[dict] = None

    persistent_workers: bool = True
    pin_memory: bool = True

    # Конфиг чекпоинтов и метрик
    checkpointing_enabled: bool = True

    # Конфиг препроцессинга
    val_resize_size: int = 512
    train_resize_size: int = 512
    val_crop_size: int = 512
    train_crop_size: int = 352  # Вычситано из соотношения для классификации
    k_folds: int = 0
    shuffle_folds: bool = True
    fast_dev_run: bool = False

class SegmentationModel(L.LightningModule):
    def __init__(self, params: Params = DEFAULT_PARAMS):
        super().__init__()

        self.save_hyperparameters()

        self.params = params

        self.model = smp.UnetPlusPlus(
            encoder_name=self.params.encoder_name,
            encoder_depth=self.params.encoder_depth,
            encoder_weights=self.params.encoder_weights,
            decoder_use_batchnorm=self.params.decoder_use_batchnorm,
            decoder_channels=self.params.decoder_channels,
            decoder_attention_type=self.params.decoder_attention_type,
            in_channels=self.params.in_channels,
            classes=self.params.num_classes,
            activation=self.params.activation,
            aux_params=self.params.aux_params,
        )

        self.multiclass_dice_loss_fn = DiceLoss(
            MULTICLASS_MODE,
            from_logits=True,  # I do not want to convert it to probabilities myself.
            log_loss=False,  # I do not want to see negative values.
            smooth=0.1,  # Leave as is.
            eps=1e-07,  # Same.
        )

        self.binary_dice_loss_fn = DiceLoss(
            BINARY_MODE,
            from_logits=True,  # I do not want to convert it to probabilities myself.
            log_loss=False,  # I do not want to see negative values.
            smooth=0.1,  # Leave as is.
            eps=1e-07,  # Same.
        )

        self.multiclass_ce_loss_fn = SoftCrossEntropyLoss(
            smooth_factor=0.1,
            reduction="mean",  # I want loss to be in [0, 1] range.
        )

        self.binary_ce_loss_fn = SoftBCEWithLogitsLoss(
            smooth_factor=0.1,
            reduction="mean",  # I want loss to be in [0, 1] range.
        )

        # This parameter define how many losses are combine to for final loss.
        # Used as division constant.
        self.__losses__ = [
            "multiclass_dice_loss",
            "tumor_dice_loss",
            "multiclass_ce_loss",
            "tumor_ce_loss"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.params.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=2e-05,
            eps=1e-07,
        )

        # Был использован для первого прогона
        self.lr_scheduler = PolynomialLR(
            self.optimizer, total_iters=self.trainer.estimated_stepping_batches, power=0.9
        )

        # self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.params.epochs, eta_min=self.params.learning_rate)

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": self.lr_scheduler, "interval": "step"},
            # "lr_scheduler": self.lr_scheduler,
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        image, mask_true, label_true = batch  # [B, 1, 512, 512] and [B, 1, 512, 512]

        assert image.ndim == 4, f"Assertion: Number of dimensions of an image must be 4"
        h, w = image.shape[2:]
        assert (
                h == 512 and w == 512
        ), f"Assertion: Height and width of an image must both be 512."

        mask_pred_logits = self(image)  # [B, 4, 512, 512] -> [B, 4, 512, 512]

        assert not torch.isnan(mask_pred_logits).any(), "Logits contain NaNs"
        assert not torch.isinf(mask_pred_logits).any(), "Logits contain Infs"
        assert not torch.isnan(mask_true).any(), "Targets contain NaNs"
        assert not torch.isinf(mask_true).any(), "Targets contain Infs"

        multiclass_ce_loss = self.multiclass_ce_loss_fn(mask_pred_logits, mask_true)
        multiclass_dice_loss = self.multiclass_dice_loss_fn(mask_pred_logits, mask_true)

        tumor_mask = torch.tensor((mask_true == 2), dtype=torch.long).unsqueeze(
            1
        )  # [B, 1, 512, 512]
        # cyst_mask = torch.tensor((mask_true == 3), dtype=torch.long).unsqueeze(
        #     1
        # )  # [B, 1, 512, 512]

        tumor_logits = mask_pred_logits[:, 2, :, :].unsqueeze(1)  # [B, 1, 512, 512]
        # cyst_logits = mask_pred_logits[:, 3, :, :].unsqueeze(1)  # [B, 1, 512, 512]

        tumor_ce_loss = self.binary_ce_loss_fn(tumor_logits, tumor_mask)
        tumor_dice_loss = self.binary_dice_loss_fn(tumor_logits, tumor_mask)
        # cyst_ce_loss = self.binary_ce_loss_fn(cyst_logits, cyst_mask)
        # cyst_dice_loss = self.binary_dice_loss_fn(cyst_logits, cyst_mask)

        loss = (
                       (multiclass_ce_loss + multiclass_dice_loss)
                       + (tumor_ce_loss + tumor_dice_loss)
                    #    + (cyst_ce_loss + cyst_dice_loss)
               ) / len(self.__losses__)

        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_lr": self.lr_scheduler.get_last_lr()[0],
                "train_ce_multiclass_loss": multiclass_ce_loss.item(),
                "train_ce_tumor_loss": tumor_ce_loss.item(),
                # "train_ce_cyst_loss": cyst_ce_loss.item(),
                "train_dice_multiclass_loss": multiclass_dice_loss.item(),
                "train_dice_tumor_loss": tumor_dice_loss.item(),
                # "train_dice_cyst_loss": cyst_dice_loss.item(),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {"loss": loss}