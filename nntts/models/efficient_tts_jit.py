from nntts.models.efficient_tts import EfficientTTSCNN
import torch


class EfficientTTSCNNJIT(EfficientTTSCNN):
    """
    docstring
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.example_input_array = torch.arange(  # TODO: change this
            self.hparams.num_symbols
        ).unsqueeze(0)
        self.eval()
        self.freeze()

    @classmethod
    def load_from_checkpoint(cls, p):
        m = super().load_from_checkpoint(p)
        m.example_input_array = torch.arange(m.hparams.num_symbols).unsqueeze(
            0
        )
        m.eval()
        m.freeze()
        return m

    def forward(self, tokens):
        mel_pred, reconst_alpha = self.inference(tokens)
        return mel_pred.transpose(1, 2), reconst_alpha
