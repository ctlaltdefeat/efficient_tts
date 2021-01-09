import warnings

warnings.simplefilter("ignore", UserWarning)

import logging

from nntts.models.efficient_tts import EfficientTTSCNN
from nntts.datasets.taco2_data import TextMelLoader, TextMelCollate
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

logging.getLogger("lightning").setLevel(logging.WARNING)

if __name__ == "__main__":
    torch.manual_seed(1234)
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
        # sigma=0.01,
        # sigma_e=0.5,
        lr=5e-4,
        eps=1e-10,
        weight_decay=1e-6,
        warmup_steps=40,
        # dropout_rate=0.0,
        # n_decoder_layer=5,
        # n_text_encoder_layer=3,
        # n_mel_encoder_layer=3,
        # n_duration_layer=2,
        # use_mse=False,
    )
    # model = EfficientTTSCNN.load_from_checkpoint(
    #     r"/workspace/efficient_tts/lightning_logs/version_26/checkpoints/last.ckpt",
    #     # lr=2e-4,
    #     # warmup_steps=40,
    #     sigma=0.01,
    #     #     #     # sigma_e=0.5,
    #     #     #     # lr=5e-4,
    #     # weight_decay=1e-6,
    #     # dropout_rate=0.5,
    #     n_decoder_layer=5,
    #     strict=False
    # )
    trainer = pl.Trainer(
        accelerator="ddp",
        gpus=-1,
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
        # plugins='ddp_sharded'
        # precision=16
        # resume_from_checkpoint=r"/workspace/efficient_tts/lightning_logs/version_11/checkpoints/last.ckpt"
        # , limit_train_batches=0.05
    )
    trainer.fit(model, train_loader, val_loader)
