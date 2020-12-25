import logging
import numpy as np
import os
import torch
import uuid
import json
from scipy.io.wavfile import write

# from ts.torch_handler.base_handler import BaseHandler
# from tacotron2_modified import Tacotron2Model
from nntts.models.efficient_tts_jit import EfficientTTSCNNJIT
from text_frontend import TextFrontend
import importlib

models = importlib.import_module("hifi-gan.models")
env = importlib.import_module("hifi-gan.env")
meldataset = importlib.import_module("hifi-gan.meldataset")


spec_gen = EfficientTTSCNNJIT.load_from_checkpoint(
    "/workspace/efficient_tts/lightning_logs/version_3/checkpoints/last.ckpt"
)
tf = TextFrontend(
    text_cleaners=["english_cleaners"],
    use_phonemes=True,
    n_jobs=1,
    with_stress=True,
    language="en-us",
)
with open("hifi-gan/config.json") as f:
    data = f.read()
h = env.AttrDict(json.loads(data))
vocoder = models.Generator(h)
vocoder.load_state_dict(
    torch.load(
        "/workspace/efficient_tts/hifi-gan/pretrained_universal/g_02500000"
    )["generator"]
)
vocoder.remove_weight_norm()
vocoder.eval()


str_input = "Hello, how are you doing?"
tokens = torch.tensor([tf.nchars] + tf(str_input)).long().unsqueeze(0)
spec, reconst_alpha = spec_gen(tokens)

audio = vocoder(spec)

audio = audio[0].squeeze() * meldataset.MAX_WAV_VALUE
audio_numpy = audio.data.cpu().numpy().astype("int16")
# path = "/tmp/{}.wav".format(uuid.uuid4().hex)
path = "sample.wav"
write(path, 22050, audio_numpy)