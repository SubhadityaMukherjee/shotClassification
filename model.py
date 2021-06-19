import albumentations as A
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import *
from helpers import *

lg = outLogger()

from efficientnet_pytorch import EfficientNet
class LitModel(pl.LightningModule):
    def __init__(self, num_classes,model, learning_rate=1e-4, weight_decay=0.0001):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.num_classes = num_classes

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = model
        
    #     @sn.snoop()

    def forward(self, x):
        out = self.model(x)
        return F.softmax(out, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        return ([optimizer], [scheduler])

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch["x"], train_batch["y"]
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        acc = accuracy(preds, y)
        self.log("train_acc_step", acc)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch["x"], val_batch["y"]
        preds = self(x)
        loss = F.cross_entropy(preds, y)
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
        train_transforms, 
        valid_transforms,
        img_size=(256, 256),
    ):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = A.Compose(
            train_transforms,
            p=1.0,
        )
        self.valid_transform = A.Compose(
            valid_transforms,
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
        lg("Finished setup")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=12, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.batch_size, num_workers=12
        )
