import warnings

warnings.simplefilter("ignore", UserWarning)

from pytorch_lightning import callbacks

import logging

from nntts.models.efficient_tts import EfficientTTSCNN
from nntts.datasets.taco2_data import TextMelLoader, TextMelCollate
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl

logging.getLogger("lightning").setLevel(logging.WARNING)

if __name__ == "__main__":
    train = TextMelLoader(r"datasets/LJSpeech-1.1/train.txt")
    train_loader = DataLoader(
        train,
        collate_fn=TextMelCollate(),
        batch_size=256,
        pin_memory=True,
        num_workers=16,
    )
    val = TextMelLoader(r"datasets/LJSpeech-1.1/val.txt")
    val_loader = DataLoader(
        val,
        collate_fn=TextMelCollate(),
        batch_size=256,
        pin_memory=True,
        num_workers=2,
    )
    model = EfficientTTSCNN(
        train.tf.nchars + 1,
        sigma=0.1,
        sigma_e=0.5,
        lr=1e-3,
        weight_decay=1e-6,
        dropout_rate=0.0,
        n_decoder_layer=7
        # n_text_encoder_layer=6
    )
    # model = EfficientTTSCNN.load_from_checkpoint(
    #     r"/workspace/efficient_tts/lightning_logs/version_5/checkpoints/last.ckpt",
    #     sigma=0.01,
    #     sigma_e=0.5,
    #     lr=5e-4,
    #     weight_decay=1e-5,
    #     dropout_rate=0.1,
    #     n_text_encoder_layer=6
    # )
    trainer = pl.Trainer(
        accelerator="ddp",
        gpus=4,
        gradient_clip_val=1,
        check_val_every_n_epoch=5,
        benchmark=True,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                monitor="val_loss", save_last=True, save_top_k=3, mode="min"
            ),
        ],
        max_epochs=3000,
        # resume_from_checkpoint=r"/workspace/efficient_tts/lightning_logs/version_3/checkpoints/last.ckpt"
        # , limit_train_batches=0.05
    )
    trainer.fit(model, train_loader, val_loader)
