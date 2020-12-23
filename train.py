from nntts.models.efficient_tts import EfficientTTSCNN
from nntts.datasets.taco2_data import TextMelLoader, TextMelCollate

# from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl

if __name__ == "__main__":
    train = TextMelLoader(r"datasets\LJSpeech-1.1\train.txt")
    train_loader = DataLoader(
        train,
        collate_fn=TextMelCollate(),
        batch_size=1,
        pin_memory=False,
        num_workers=0,
    )
    val = TextMelLoader(r"datasets\LJSpeech-1.1\val.txt")
    val_loader = DataLoader(
        val,
        collate_fn=TextMelCollate(),
        batch_size=1,
        pin_memory=False,
        num_workers=0,
    )
    model = EfficientTTSCNN(
        train.tf.nchars + 1,
        n_mel_encoder_layer=1,
        n_text_encoder_layer=1,
        n_decoder_layer=1,
        n_channels=64,
        symbol_embedding_dim=64,
    )
    trainer = pl.Trainer(
        gpus=0, gradient_clip_val=1, benchmark=False, limit_train_batches=0.05
    )
    trainer.fit(model, train_loader, val_loader)
