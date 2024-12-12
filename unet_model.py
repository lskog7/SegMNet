# |-----------------------------------------------------------------------|
# | BASE CLASS FOR UNET++ SEGMENTATION MODEL WITH EFFICIENTNET-B0 ENCODER |
# |-----------------------------------------------------------------------|

# |------------------------------------------------------------------|
# | Description:                                                     |
# |------------------------------------------------------------------|
# | Author: Artemiy Tereshchenko                                     |
# |------------------------------------------------------------------|

# Libraries:
import torch
from typing import Optional
import pytorch_lightning as L
from segmentation_models_pytorch import UnetPlusPlus
from segmentation_models_pytorch.losses import (
    DiceLoss,
    MULTICLASS_MODE,
    BINARY_MODE,
    SoftBCEWithLogitsLoss,
    SoftCrossEntropyLoss,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from typing import List
import logging

# Local modules:
...

# Module-specific logging template:
logging.basicConfig(level=logging.INFO, format="MODULE->[unet_model.py]: %(message)s")

class UNet(L.LightningModule):
    # I do not recommend to change any default values.
    # They are used for loading model from checkpoint and can lead to unexpected errors.
    # Generally, I already tuned these parameters and left them changable just not to rewrite code.
    def __init__(
        self,
        encoder_name: str = "efficientnet-b0",  # Can be: {"mobilenet_v2": "2M", "efficientnet-b5": "28M"}
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_use_batchnorm: bool = True, # True, False, "inplace"
        decoder_attention_type: str | None = None, # None or "scse"
        in_channels: int = 1,
        classes: int = 4,
        activation: Optional[str] = None,
        aux_params: Optional[dict] = None,
        lr: float = 1e-06,
        smooth: float = 1e-06,
        eps: float = 1e-06,
    ):
        super().__init__()
        """
            Unet model for semantic segmentation.

            Args:
                encoder_name (str): Name of the encoder to be used in the model.
                encoder_depth (int): Depth of the encoder.
                encoder_weights (str): Pre-trained weights for the encoder.
                encoder_output_stride (int): Output stride of the encoder.
                decoder_channels (int): Number of channels in the decoder.
                decoder_atrous_rates (tuple): Atrous rates for the decoder.
                in_channels (int): Number of input channels.
                classes (int): Number of classes in the output.
                activation (str): Activation function to be used.
                upsampling (int): Upsampling factor for the output.
                aux_params (dict): Additional parameters for auxiliary tasks.
                lr (float): Learning rate for the optimizer.
                smooth (float): Smooth factor for loss functions.
                eps (float): Epsilon value for Dice Loss.
                out_threshold (float): Threshold value for binary classification.
            """

        # |-----------------------------------|
        # | DEFINITIONS OF INTERNAL VARIABLES |
        # |-----------------------------------|

        # Save model hyperparameters.
        self.save_hyperparameters()

        # Define all the local variables.
        self.encoder_name = encoder_name
        self.encoder_depth = encoder_depth
        self.encoder_weights = encoder_weights
        self.decoder_use_batchnorm = decoder_use_batchnorm
        self.decoder_channels = decoder_channels
        self.decoder_attention_type = decoder_attention_type
        self.in_channels = in_channels
        self.classes = classes
        self.activation = activation
        self.aux_params = aux_params
        self.learning_rate = lr
        self.smooth = smooth
        self.epsilon = eps

        # Define DeepLabV3 model itself.
        # I use models from SMP.
        self.model = UnetPlusPlus(
            encoder_name=self.encoder_name,
            encoder_depth=self.encoder_depth,
            encoder_weights=self.encoder_weights,
            decoder_use_batchnorm=self.decoder_use_batchnorm,
            decoder_channels=self.decoder_channels,
            decoder_attention_type=self.decoder_attention_type,
            in_channels=self.in_channels,
            classes=self.classes,
            activation=self.activation,
            aux_params=self.aux_params,
        )

        # A friendly reminder.
        # Loss multiclass mode suppose you are solving multi-class segmentation task. That mean you have C = 1..N classes which have unique label values, classes are mutually exclusive and all pixels are labeled with theese values. Target mask shape - (B, H, W), model output mask shape (B, C, H, W).

        # Define loss functions:
        # First, multicalss DiceLoss for general case with 4 classes.
        # y_pred - torch.Tensor of shape (B, C, H, W)
        # y_true - torch.Tensor of shape (B, H, W) or (B, C, H, W)
        self.multiclass_dice_loss_fn = DiceLoss(
            MULTICLASS_MODE,
            from_logits=True,  # I do not want to convert it to probabilities myself.
            log_loss=False,  # I do not want to see negative values.
            smooth=self.smooth,  # Leave as is.
            eps=self.epsilon,  # Same.
        )

        # Second, binary DiceLoss for tumor and cyst segmneation task.
        # Got to add this, because in other case, there will be no good tumor predictions.
        # y_pred - torch.Tensor of shape (B, C, H, W)
        # y_true - torch.Tensor of shape (B, H, W) or (B, C, H, W)
        self.binary_dice_loss_fn = DiceLoss(
            BINARY_MODE,
            from_logits=True,  # I do not want to convert it to probabilities myself.
            log_loss=False,  # I do not want to see negative values.
            smooth=self.smooth,  # Leave as is.
            eps=self.epsilon,  # Same.
        )

        # Third, CrossEnthropyLoss for multiclass segmentation.
        # It is broadly used in medical image segmentation.
        # y_pred - torch.Tensor of shape (B, C, H, W)
        # y_true - torch.Tensor of shape (B, H, W)
        self.multiclass_ce_loss_fn = SoftCrossEntropyLoss(
            smooth_factor=self.smooth,
            reduction="mean",  # I want loss to be in [0, 1] range.
        )

        # Fourth, BinaryCrossEnthropyLoss for binary segmentation.
        # Also for better tumor and cyst segmentation.
        # y_pred - torch.Tensor of shape (B, C, H, W)
        # y_true - torch.Tensor of shape (B, H, W) or (B, 1 ,H, W)
        self.binary_ce_loss_fn = SoftBCEWithLogitsLoss(
            smooth_factor=self.smooth,
            reduction="mean",  # I want loss to be in [0, 1] range.
        )

    # |-----------------------------------------------|
    # | DEFINITIONS OF INTERNAL METHODS AND FUNCTIONS |
    # |-----------------------------------------------|

    # First step, define a forward pass. Just to use self.model(x) in future.
    # Also, i define types of inputs and outputs where it is possible.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    # Second step, define optimizer and LR scheduler for the model.
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Using AdamW optimizer as it is among the most efficient ones.
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        # For the scheduler I use OneCycleLR.
        # It has nice LR curve and it is easy to use.
        # self.lr_scheduler = OneCycleLR(
        #     self.optimizer,
        #     max_lr=self.learning_rate,
        #     total_steps=self.trainer.estimated_stepping_batches,  # Just do not forget to define the total number of steps.
        #     div_factor=25,  # Initial LR is self.learning_rate / div_factor.
        #     final_div_factor=10000,  # Final LR is self.learning_rate / final_div_factor.
        # )
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.75, patience=10, min_lr=1e-07)

        # Do not change the form of the dictionary.
        # It can cause errors and they are very time-consuming.
        # return {
        #     "optimizer": self.optimizer,
        #     "lr_scheduler": {"scheduler": self.lr_scheduler, "interval": "step"},
        # }
        return {"optimizer": self.optimizer,
                "lr_scheduler": {"scheduler": self.lr_scheduler, "monitor":"val_loss"}}

    # Now, begin training operations themselves.
    # First, define a training step.
    # (To be honest, other ones are very similar to this one, comments are nearly the same. I copied them.)
    # Generally, it does not has to return anything, but it can.
    # I use dictionary with loss.
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Training step.

        Definitions:
          1. x == image [B, 1, 512, 512] | Values range: [0, 1]
          2. y == mask [B, 1, 512, 512] | Values: [0, 1, 2, 3]
        I just use "x" and "y" for simplicity. But you can use "image" and "mask".
        Pay attention to size of "x" and "y". They must be the same (512).
        x, y_true = batch  # [B, 1, 512, 512] and [B, 1, 512, 512]
        """
        # Definitions:
        #   1. x == image [B, 1, 512, 512] | Values range: [0, 1]
        #   2. y == mask [B, 1, 512, 512] | Values: [0, 1, 2, 3]
        # I just use "x" and "y" for simplicity. But you can use "image" and "mask".
        # Pay attention to size of "x" and "y". They must be the same (512).
        x, y_true = batch  # [B, 1, 512, 512] and [B, 1, 512, 512]

        # Two basic assertions:
        #   1. Number of dimensions of an image must be 4. [B, 1, 512, 512]
        #   2. Height and width of an image must both be 512.
        assert x.ndim == 4, f"Assertion: Number of dimensions of an image must be 4"
        h, w = x.shape[2:]
        assert (
            h == 512 and w == 512
        ), f"Assertion: Height and width of an image must both be 512."

        # Calcuate logits with forward pass.
        # And also calculate the prediction itself.
        logits = self(x)  # [B, 4, 512, 512] -> [B, 4, 512, 512]
        # y_pred = logits.argmax(dim=1)  # [B, 4, 512, 512] -> [B, 1, 512, 512]

        # Change the mask dtype to torch.long (e.g. int). To match the outputs of the model.
        y_true = y_true.long()  # [B, 1, 512, 512] -> [B, 1, 512, 512]

        # Another block of assertions, we need to make sure that no NaNs or Infs are present.
        assert not torch.isnan(logits).any(), "Logits contain NaNs"
        assert not torch.isinf(logits).any(), "Logits contain Infs"
        assert not torch.isnan(y_true).any(), "Targets contain NaNs"
        assert not torch.isinf(y_true).any(), "Targets contain Infs"

        # Calculate all losses:
        #   1. Multiclass CE.
        #   2. Multicalss DiceLoss.
        #   3. Binary CE for tumors (label == 2).
        #   4. Binary DiceLoss for tumors (label == 2).
        #   5. Binary CE for cysts (label == 3).
        #   6. Binary DiceLoss for cysts (label == 3).
        # Multiclass case is quite simple. You do no need to change anything.
        multiclass_ce_loss = self.multiclass_ce_loss_fn(logits, y_true)
        multiclass_dice_loss = self.multiclass_dice_loss_fn(logits, y_true)

        # Binary case is a bit more complicated:
        #   - You need to choose appropriate part of the y_true and y_pred.
        #   - To choose tumor part:
        #       1. Choose y_true where it is 2.
        #       2. Choose the second layer of logits.
        # I do those transforms explicitly. The do not consume much memory.
        tumor_mask = (y_true == 2).long()  # [B, 1, 512, 512]
        cyst_mask = (y_true == 3).long()  # [B, 1, 512, 512]

        # Here you need to use x.unsqueeze(1) to add a dimension.
        # This it beacause loss functions need predictions to have shape of [B, C, H, W].
        tumor_logits = logits[:, 2, :, :].unsqueeze(1)  # [B, 1, 512, 512]
        cyst_logits = logits[:, 3, :, :].unsqueeze(1)  # [B, 1, 512, 512]

        # Now it is time to calculate binary losses.
        tumor_ce_loss = self.binary_ce_loss_fn(tumor_logits, tumor_mask)
        tumor_dice_loss = self.binary_dice_loss_fn(tumor_logits, tumor_mask)
        cyst_ce_loss = self.binary_ce_loss_fn(cyst_logits, cyst_mask)
        cyst_dice_loss = self.binary_dice_loss_fn(cyst_logits, cyst_mask)

        # Calculate total loss for this step.
        # To be honest, I want my loss to be in range [0, 1]. But in this case, I do not do this.
        # That is beacause I add coeeficients to losses.
        # Loss consists of three parts and lies in interval [0, 8]:
        loss = (
            2.0 * (multiclass_ce_loss + multiclass_dice_loss)  # Multiclass part.
            + 4.0 * (tumor_ce_loss + tumor_dice_loss)  # Tumor part.
            + 1.0 * (cyst_ce_loss + cyst_dice_loss)  # Cyst part.
        ) / 14.0

        # Here is a block of logging code.
        # That is necessary to see the progress of training in tensorboard.
        # I log nearly everything both on step and epoch.
        # You can change it if you want.
        # First, two main metrics:
        #   1. LOSS
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   2. LEARNING RATE
        self.log(
            "train_lr",
            self.lr_scheduler.get_last_lr()[0],
            on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Second, additional matrics:
        #   1. MULTICLASS DICE
        self.log(
            "train_multiclass_dice_loss",
            multiclass_dice_loss,
            on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   2. MULTICLASS CE
        self.log(
            "train_multiclass_ce_loss",
            multiclass_ce_loss,
            on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   3. BINARY TUMOR DICE
        self.log(
            "train_tumor_dice_loss",
            tumor_dice_loss,
            on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   4. BINARY TUMOR CE
        self.log(
            "train_tumor_ce_loss",
            tumor_ce_loss,
            on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   5. BINARY CYSTS DICE
        self.log(
            "train_cyst_dice_loss",
            cyst_dice_loss,
            on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   6. BINARY CYSTS CE
        self.log(
            "train_cyst_ce_loss",
            cyst_ce_loss,
            on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {"loss": loss}

    # Validation step is totaly similar to training step.
    # I just copied all the code from training step here.
    # Nothing new.
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Validation step.

        Definitions:
          1. x == image [B, 1, 512, 512] | Values range: [0, 1]
          2. y == mask [B, 1, 512, 512] | Values: [0, 1, 2, 3]
        I just use "x" and "y" for simplicity. But you can use "image" and "mask".
        Pay attention to size of "x" and "y". They must be the same (512).
        x, y_true = batch  # [B, 1, 512, 512] and [B, 1, 512, 512]
        """
        # Definitions:
        #   1. x == image [B, 1, 512, 512] | Values range: [0, 1]
        #   2. y == mask [B, 1, 512, 512] | Values: [0, 1, 2, 3]
        # I just use "x" and "y" for simplicity. But you can use "image" and "mask".
        # Pay attention to size of "x" and "y". They must be the same (512).
        x, y_true = batch  # [B, 1, 512, 512] and [B, 1, 512, 512]

        # Two basic assertions:
        #   1. Number of dimensions of an image must be 4. [B, 1, 512, 512]
        #   2. Height and width of an image must both be 512.
        assert x.ndim == 4, f"Assertion: Number of dimensions of an image must be 4"
        h, w = x.shape[2:]
        assert (
            h == 512 and w == 512
        ), f"Assertion: Height and width of an image must both be 512."

        # Calcuate logits with forward pass.
        # And also calculate the prediction itself.
        logits = self(x)  # [B, 4, 512, 512] -> [B, 4, 512, 512]
        # y_pred = logits.argmax(dim=1)  # [B, 4, 512, 512] -> [B, 1, 512, 512]

        # Change the mask dtype to torch.long (e.g. int). To match the outputs of the model.
        # y_true = y_true.long()  # [B, 1, 512, 512] -> [B, 1, 512, 512]

        # Another block of assertions, we need to make sure that no NaNs or Infs are present.
        assert not torch.isnan(logits).any(), "Logits contain NaNs"
        assert not torch.isinf(logits).any(), "Logits contain Infs"
        assert not torch.isnan(y_true).any(), "Targets contain NaNs"
        assert not torch.isinf(y_true).any(), "Targets contain Infs"

        # Calculate all losses:
        #   1. Multiclass CE.
        #   2. Multicalss DiceLoss.
        #   3. Binary CE for tumors (label == 2).
        #   4. Binary DiceLoss for tumors (label == 2).
        #   5. Binary CE for cysts (label == 3).
        #   6. Binary DiceLoss for cysts (label == 3).
        # Multiclass case is quite simple. You do no need to change anything.
        multiclass_ce_loss = self.multiclass_ce_loss_fn(logits, y_true)
        multiclass_dice_loss = self.multiclass_dice_loss_fn(logits, y_true)

        # Binary case is a bit more complicated:
        #   - You need to choose appropriate part of the y_true and y_pred.
        #   - To choose tumor part:
        #       1. Choose y_true where it is 2.
        #       2. Choose the second layer of logits.
        # I do those transforms explicitly. They do not consume much memory.
        tumor_mask = (y_true == 2).long()  # [B, 1, 512, 512]
        cyst_mask = (y_true == 3).long()  # [B, 1, 512, 512]

        # Here you need to use x.unsqueeze(1) to add a dimension.
        # This it beacause loss functions need predictions to have shape of [B, C, H, W].
        tumor_logits = logits[:, 2, :, :].unsqueeze(1)  # [B, 1, 512, 512]
        cyst_logits = logits[:, 3, :, :].unsqueeze(1)  # [B, 1, 512, 512]

        # Now it is time to calculate binary losses.
        tumor_ce_loss = self.binary_ce_loss_fn(tumor_logits, tumor_mask)
        tumor_dice_loss = self.binary_dice_loss_fn(tumor_logits, tumor_mask)
        cyst_ce_loss = self.binary_ce_loss_fn(cyst_logits, cyst_mask)
        cyst_dice_loss = self.binary_dice_loss_fn(cyst_logits, cyst_mask)

        # Calculate total loss for this step.
        # To be honest, I want my loss to be in range [0, 1]. But in this case, I do not do this.
        # That is beacause I add coeeficients to losses.
        # Loss consists of three parts and lies in interval [0, 8]:
        loss = (
            2.0 * (multiclass_ce_loss + multiclass_dice_loss)  # Multiclass part.
            + 4.0 * (tumor_ce_loss + tumor_dice_loss)  # Tumor part.
            + 1.0 * (cyst_ce_loss + cyst_dice_loss)  # Cyst part.
        ) / 14.0

        # Here is a block of logging code.
        # That is necessary to see the progress of training in tensorboard.
        # I log nearly everything both on step and epoch.
        # You can change it if you want.
        # First, two main metrics:
        #   1. LOSS
        self.log(
            "val_loss",
            loss,
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   2. LEARNING RATE
        self.log(
            "val_lr",
            self.lr_scheduler.get_last_lr()[0],
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Second, additional matrics:
        #   1. MULTICLASS DICE
        self.log(
            "val_multiclass_dice_loss",
            multiclass_dice_loss,
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   2. MULTICLASS CE
        self.log(
            "val_multiclass_ce_loss",
            multiclass_ce_loss,
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   3. BINARY TUMOR DICE
        self.log(
            "val_tumor_dice_loss",
            tumor_dice_loss,
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   4. BINARY TUMOR CE
        self.log(
            "val_tumor_ce_loss",
            tumor_ce_loss,
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   5. BINARY CYSTS DICE
        self.log(
            "val_cyst_dice_loss",
            cyst_dice_loss,
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   6. BINARY CYSTS CE
        self.log(
            "val_cyst_ce_loss",
            cyst_ce_loss,
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {"loss": loss}

    # Test step is also similar to training step and validation step.
    # I also copied everything, thanks to Lightning syntax.
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Test step.

        Definitions:
          1. x == image [B, 1, 512, 512] | Values range: [0, 1]
          2. y == mask [B, 1, 512, 512] | Values: [0, 1, 2, 3]
        I just use "x" and "y" for simplicity. But you can use "image" and "mask".
        Pay attention to size of "x" and "y". They must be the same (512).
        x, y_true = batch  # [B, 1, 512, 512] and [B, 1, 512, 512]
        """
        # Definitions:
        #   1. x == image [B, 1, 512, 512] | Values range: [0, 1]
        #   2. y == mask [B, 1, 512, 512] | Values: [0, 1, 2, 3]
        # I just use "x" and "y" for simplicity. But you can use "image" and "mask".
        # Pay attention to size of "x" and "y". They must be the same (512).
        x, y_true = batch  # [B, 1, 512, 512] and [B, 1, 512, 512]

        # Two basic assertions:
        #   1. Number of dimensions of an image must be 4. [B, 1, 512, 512]
        #   2. Height and width of an image must both be 512.
        assert x.ndim == 4, f"Assertion: Number of dimensions of an image must be 4"
        h, w = x.shape[2:]
        assert (
            h == 512 and w == 512
        ), f"Assertion: Height and width of an image must both be 512."

        # Calcuate logits with forward pass.
        # And also calculate the prediction itself.
        logits = self(x)  # [B, 4, 512, 512] -> [B, 4, 512, 512]
        # y_pred = logits.argmax(dim=1)  # [B, 4, 512, 512] -> [B, 1, 512, 512]

        # Change the mask dtype to torch.long (e.g. int). To match the outputs of the model.
        y_true = y_true.long()  # [B, 1, 512, 512] -> [B, 1, 512, 512]

        # Another block of assertions, we need to make sure that no NaNs or Infs are present.
        assert not torch.isnan(logits).any(), "Logits contain NaNs"
        assert not torch.isinf(logits).any(), "Logits contain Infs"
        assert not torch.isnan(y_true).any(), "Targets contain NaNs"
        assert not torch.isinf(y_true).any(), "Targets contain Infs"

        # Calculate all losses:
        #   1. Multiclass CE.
        #   2. Multicalss DiceLoss.
        #   3. Binary CE for tumors (label == 2).
        #   4. Binary DiceLoss for tumors (label == 2).
        #   5. Binary CE for cysts (label == 3).
        #   6. Binary DiceLoss for cysts (label == 3).
        # Multiclass case is quite simple. You do no need to change anything.
        multiclass_ce_loss = self.multiclass_ce_loss_fn(logits, y_true)
        multiclass_dice_loss = self.multiclass_dice_loss_fn(logits, y_true)

        # Binary case is a bit more complicated:
        #   - You need to choose appropriate part of the y_true and y_pred.
        #   - To choose tumor part:
        #       1. Choose y_true where it is 2.
        #       2. Choose the second layer of logits.
        # I do those transforms explicitly. The do not consume much memory.
        tumor_mask = (y_true == 2).long()  # [B, 1, 512, 512]
        cyst_mask = (y_true == 3).long()  # [B, 1, 512, 512]

        # Here you need to use x.unsqueeze(1) to add a dimension.
        # This it beacause loss functions need predictions to have shape of [B, C, H, W].
        tumor_logits = logits[:, 2, :, :].unsqueeze(1)  # [B, 1, 512, 512]
        cyst_logits = logits[:, 3, :, :].unsqueeze(1)  # [B, 1, 512, 512]

        # Now it is time to calculate binary losses.
        tumor_ce_loss = self.binary_ce_loss_fn(tumor_logits, tumor_mask)
        tumor_dice_loss = self.binary_dice_loss_fn(tumor_logits, tumor_mask)
        cyst_ce_loss = self.binary_ce_loss_fn(cyst_logits, cyst_mask)
        cyst_dice_loss = self.binary_dice_loss_fn(cyst_logits, cyst_mask)

        # Calculate total loss for this step.
        # To be honest, I want my loss to be in range [0, 1]. But in this case, I do not do this.
        # That is beacause I add coeeficients to losses.
        # Loss consists of three parts and lies in interval [0, 8]:
        loss = (
            2.0 * (multiclass_ce_loss + multiclass_dice_loss)  # Multiclass part.
            + 4.0 * (tumor_ce_loss + tumor_dice_loss)  # Tumor part.
            + 1.0 * (cyst_ce_loss + cyst_dice_loss)  # Cyst part.
        ) / 14.0

        # Here is a block of logging code.
        # That is necessary to see the progress of training in tensorboard.
        # I log nearly everything both on step and epoch.
        # You can change it if you want.
        # First, two main metrics:
        #   1. LOSS
        self.log(
            "test_loss",
            loss,
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   2. LEARNING RATE
        # self.log(
        #     "test_lr",
        #     self.lr_scheduler.get_last_lr()[0],
        #     # on_step=True,
        #     on_epoch=True,
        #     # prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        # )

        # Second, additional matrics:
        #   1. MULTICLASS DICE
        self.log(
            "test_multiclass_dice_loss",
            multiclass_dice_loss,
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   2. MULTICLASS CE
        self.log(
            "test_multiclass_ce_loss",
            multiclass_ce_loss,
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   3. BINARY TUMOR DICE
        self.log(
            "test_tumor_dice_loss",
            tumor_dice_loss,
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   4. BINARY TUMOR CE
        self.log(
            "test_tumor_ce_loss",
            tumor_ce_loss,
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   5. BINARY CYSTS DICE
        self.log(
            "test_cyst_dice_loss",
            cyst_dice_loss,
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        #   6. BINARY CYSTS CE
        self.log(
            "test_cyst_ce_loss",
            cyst_ce_loss,
            # on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {"loss": loss}

    # Here is a place for a predict_step function.
    # It is not implemeted, because I do not need it in inference, but you can do it yourself if you want.
