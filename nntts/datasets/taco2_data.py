import random
import numpy as np
import torch
import torch.utils.data

# from utils import load_wav_to_torch, load_filepaths_and_text
from nntts.text import text_to_sequence
from nntts.datasets.meldataset import load_wav, normalize, mel_spectrogram
from text_frontend import TextFrontend
from pathlib import Path
import dill
from text_frontend import TextFrontend


# def load_filepaths_and_text(filename, split="|"):
#     with open(filename, encoding="utf-8") as f:
#         filepaths_and_text = [line.strip().split(split) for line in f]
#     return filepaths_and_text


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(
        self,
        meta_file,
        #  text_cleaners=['english_cleaners'],
        max_wav_value=32768.0,
        sampling_rate=22050,
        lang="en-us"
        #  wav_path="/home/shaunxliu/data/LJSpeech-1.1/wavs",
        #  use_phnseq=False,
        #  phnset_path=None,
    ):
        self.tf = TextFrontend(
            text_cleaners=["english_cleaners"],
            use_phonemes=True,
            n_jobs=1,
            with_stress=True,
            language=lang,
        )
        # self.audiopaths_and_text = load_filepaths_and_text(meta_file)
        p = Path(meta_file).resolve()
        if p.with_suffix(".pkl").exists():
            self.collection = dill.load(open(p.with_suffix(".pkl"), "rb"))
        else:
            lines = open(meta_file, encoding="utf8").read().splitlines()
            self.collection = [
                (l.split("|")[0], self.tf(l.split("|")[1])) for l in lines
            ]
            dill.dump(
                self.collection, open(p.with_suffix(".pkl"), "wb"),
            )
        # self.text_cleaners = text_cleaners
        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        # self.wav_path = wav_path
        # self.use_phnseq = use_phnseq
        # if self.use_phnseq:
        #     assert phnset_path is not None, \
        #         "Please provide phnset_path if want to use phone seq as input"
        #     with open(phnset_path, "r") as f:
        #         phn_list = [l.strip() for l in f]
        #         self.phn2idx = dict(zip(phn_list, range(len(phn_list))))
        random.seed(1234)
        # random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audio_path, text_tokens):
        text = torch.tensor([self.tf.nchars] + text_tokens).long()
        mel = self.get_mel(audio_path)
        return (text, mel)

    def get_mel(self, filename):
        # wav_fid = filename.split("/")[-1]
        # wav_path = f"{self.wav_path}/{wav_fid}"
        wav_path = filename
        audio, sr = load_wav(wav_path)
        audio = audio / self.max_wav_value
        # audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        melspec = mel_spectrogram(audio)
        # print(melspec.shape)
        return melspec.squeeze(0)

    # def get_text(self, text):
    #     if self.use_phnseq:
    #         text_norm = torch.LongTensor(
    #             [self.phn2idx[p] for p in text.split()])
    #     else:
    #         text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
    #     return []

    def __getitem__(self, index):
        sample = self.collection[index]
        return self.get_mel_text_pair(sample[0], sample[1])

    def __len__(self):
        return len(self.collection)


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0,
            descending=True,
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step
                - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)

        return (
            text_padded,
            input_lengths,
            mel_padded.transpose(1, 2),
            output_lengths,
        )
