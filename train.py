import pytorch_lightning as pl
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.loggers import *


def run_training(n_epochs, model, dm, logger=CSVLogger("logs")):
    trainer = pl.Trainer(
        auto_select_gpus=True,
        gpus=1,
        precision=16,
        profiler=False,
        max_epochs=n_epochs,
        callbacks=[
            pl.callbacks.ProgressBar(),
            pl.callbacks.GPUStatsMonitor(),
            PrintTableMetricsCallback(),
        ],
        logger=logger,
        #  accelerator="ddp",
        #  plugins="ddp_sharded",
    )

    trainer.fit(model, dm)
    trainer.save_checkpoint("./logs/model1.ckpt")
    return trainer
