import albumentations as A
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import *


def lin_comb(v1, v2, beta):
    """
    Linear Combination
    """
    return beta * v1 + (1 - beta) * v2


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε: float = 0.1, reduction="mean"):
        super().__init__()
        self.ε, self.reduction = ε, reduction

    def forward(self, output, target):
        target = target.to(torch.long)
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return {"loss": lin_comb(loss / c, nll, self.ε)}


class LitModel(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        model,
        batch_size=128,
        learning_rate=1e-4,
        weight_decay=0.0001,
    ):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.num_classes = num_classes

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.model = model

    #     @sn.snoop()

    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        return ([optimizer], [scheduler])

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch["x"], train_batch["y"]
        preds = self(x)
        #  loss = F.binary_cross_entropy_with_logits(preds, y)
        loss = F.nll_loss(preds, y)
        #  loss = LabelSmoothingCrossEntropy(preds, y)
        acc = accuracy(preds, y)
        self.log("train_acc_step", acc)
        self.log("train_loss", loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch["x"], test_batch["y"]
        preds = self(x)

        loss = F.nll_loss(preds, y)
        #  loss = LabelSmoothingCrossEntropy(preds, y)
        acc = accuracy(preds, y)
        self.log("test_acc_step", acc)
        self.log("test_loss", loss)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch["x"], val_batch["y"]
        preds = self(x)
        loss = F.nll_loss(preds, y)
        #  loss = LabelSmoothingCrossEntropy(preds, y)
        acc = accuracy(preds, y)
        self.log("val_acc_step", acc)
        self.log("val_loss", loss)


class ImDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df,
        batch_size,
        num_classes,
        data_dir,
        img_size=(256, 256),
    ):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = A.Compose(
            [
                #  A.RandomResizedCrop(img_size, img_size, p=1.0),
                A.Resize(img_size, img_size),
                #  A.Transpose(p=0.5),
                #  A.HorizontalFlip(p=0.5),
                #  A.VerticalFlip(p=0.5),
                #  A.ShiftScaleRotate(p=0.5),
                #  A.HueSaturationValue(
                #      hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                #  ),
                #  A.RandomBrightnessContrast(
                #      brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                #  ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                #  A.CoarseDropout(p=0.5),
                #  A.Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
        self.valid_transform = A.Compose(
            [
                #  A.CenterCrop(img_size, img_size, p=1.0),
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )

    def setup(self, stage=None):
        dfx = pd.read_csv("./train_folds.csv")
        train = dfx.loc[dfx["kfold"] != 1]
        val = dfx.loc[dfx["kfold"] == 1]

        self.train_dataset = ImageClassDs(
            train, self.data_dir, train=True, transforms=self.train_transform
        )

        self.valid_dataset = ImageClassDs(
            val, self.data_dir, train=False, transforms=self.valid_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=12, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.batch_size, num_workers=12
        )
