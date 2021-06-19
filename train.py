import pytorch_lightning as pl
from pytorch_lightning.loggers import *


def run_training(n_epochs, model, dm, logger=CSVLogger("logs")):
    trainer = pl.Trainer(
        auto_select_gpus=True,
        gpus=1,
        precision=16,
        profiler=False,
        max_epochs=n_epochs,
        callbacks=[pl.callbacks.ProgressBar()],
        logger=logger,
        accelerator="ddp",
        plugins="ddp_sharded",
    )

    trainer.fit(model, dm)
    trainer.save_checkpoint("model1.ckpt")
