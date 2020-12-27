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
        batch_size=32,
        pin_memory=True,
        num_workers=8,
    )
    val = TextMelLoader(r"datasets/LJSpeech-1.1/val.txt")
    val_loader = DataLoader(
        val,
        collate_fn=TextMelCollate(),
        batch_size=32,
        pin_memory=True,
        num_workers=2,
    )
    model = EfficientTTSCNN(
        train.tf.nchars + 1,
        sigma=0.05,
        #     #     sigma_e=0.5,
        lr=2e-4,
        warmup_steps=40,
        weight_decay=1e-5,
        dropout_rate=0.1,
        n_decoder_layer=8
        #     #     # n_text_encoder_layer=6
    )
    # model = EfficientTTSCNN.load_from_checkpoint(
    #     r"/workspace/efficient_tts/lightning_logs/version_14/checkpoints/epoch=619-step=62619.ckpt",
    #     #     lr=5e-5,
    #     #     # warmup_steps=2
    #     sigma=0.01,
    #     #     #     # sigma_e=0.5,
    #     #     #     # lr=5e-4,
    #     #     #     # weight_decay=1e-5,
    #     # dropout_rate=0.0,
    #     #     #     # n_text_encoder_layer=6
    # )
    trainer = pl.Trainer(
        accelerator="ddp",
        gpus=4,
        gradient_clip_val=1,
        check_val_every_n_epoch=10,
        benchmark=True,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                monitor="val/loss", save_last=True, save_top_k=3, mode="min"
            ),
        ],
        max_epochs=5000,
        # precision=16
        # resume_from_checkpoint=r"/workspace/efficient_tts/lightning_logs/version_11/checkpoints/last.ckpt"
        # , limit_train_batches=0.05
    )
    trainer.fit(model, train_loader, val_loader)
