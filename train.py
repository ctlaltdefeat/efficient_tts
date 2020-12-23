from nntts.models.efficient_tts import EfficientTTSCNN
from nntts.datasets.taco2_data import TextMelLoader, TextMelCollate
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl

if __name__ == "__main__":
    model = EfficientTTSCNN(
        237,
        # n_mel_encoder_layer=1,
        # n_text_encoder_layer=1,
        # n_decoder_layer=1,
        # n_channels=64,
        # symbol_embedding_dim=64,
    )
    train = TextMelLoader(r"datasets\LJSpeech-1.1\train.txt")
    data_loader = DataLoader(
        train,
        collate_fn=TextMelCollate(),
        batch_size=4,
        pin_memory=True,
        num_workers=0,
    )
    trainer = pl.Trainer(gpus=1, gradient_clip_val=1, benchmark=False)
    trainer.fit(model, data_loader)
