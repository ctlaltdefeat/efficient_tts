import librosa
import matplotlib.pylab as plt
import numpy as np
import torch
from numpy import ndarray
from pytorch_lightning.utilities import rank_zero_only


def griffin_lim(magnitudes, n_iters=50, n_fft=1024):
    """
    Griffin-Lim algorithm to convert magnitude spectrograms to audio signals
    """
    phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
    complex_spec = magnitudes * phase
    signal = librosa.istft(complex_spec)
    if not np.isfinite(signal).all():
        logging.warning("audio was not finite, skipping audio saving")
        return np.array([0])

    for _ in range(n_iters):
        _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft))
        complex_spec = magnitudes * phase
        signal = librosa.istft(complex_spec)
    return signal


@rank_zero_only
def log_audio_to_tb(
    swriter,
    spect,
    name,
    step,
    griffin_lim_mag_scale=1024,
    griffin_lim_power=1.2,
    sr=22050,
    n_fft=1024,
    n_mels=80,
    fmax=8000,
):
    filterbank = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax
    )
    log_mel = spect
    mel = np.exp(log_mel)
    magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
    audio = griffin_lim(magnitude.T ** griffin_lim_power)
    swriter.add_audio(name, audio / max(np.abs(audio)), step, sample_rate=sr)


# @rank_zero_only
# def tacotron2_log_to_tb_func(
#     swriter,
#     tensors,
#     step,
#     tag="train",
#     log_images=False,
#     log_images_freq=1,
#     add_audio=True,
#     griffin_lim_mag_scale=1024,
#     griffin_lim_power=1.2,
#     sr=22050,
#     n_fft=1024,
#     n_mels=80,
#     fmax=8000,
# ):
#     _, spec_target, mel_postnet, gate, gate_target, alignments = tensors
#     if log_images and step % log_images_freq == 0:
#         swriter.add_image(
#             f"{tag}_alignment",
#             plot_alignment_to_numpy(alignments[0].data.cpu().numpy().T),
#             step,
#             dataformats="HWC",
#         )
#         swriter.add_image(
#             f"{tag}_mel_target",
#             plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()),
#             step,
#             dataformats="HWC",
#         )
#         swriter.add_image(
#             f"{tag}_mel_predicted",
#             plot_spectrogram_to_numpy(mel_postnet[0].data.cpu().numpy()),
#             step,
#             dataformats="HWC",
#         )
#         if add_audio:
#             filterbank = librosa.filters.mel(
#                 sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax
#             )
#             log_mel = mel_postnet[0].data.cpu().numpy().T
#             mel = np.exp(log_mel)
#             magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
#             audio = griffin_lim(magnitude.T ** griffin_lim_power)
#             swriter.add_audio(
#                 f"audio/{tag}_predicted",
#                 audio / max(np.abs(audio)),
#                 step,
#                 sample_rate=sr,
#             )

#             log_mel = spec_target[0].data.cpu().numpy().T
#             mel = np.exp(log_mel)
#             magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
#             audio = griffin_lim(magnitude.T ** griffin_lim_power)
#             swriter.add_audio(
#                 f"audio/{tag}_target",
#                 audio / max(np.abs(audio)),
#                 step,
#                 sample_rate=sr,
#             )


# def plot_alignment_to_numpy(alignment, info=None):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     im = ax.imshow(
#         alignment, aspect="auto", origin="lower", interpolation="none"
#     )
#     fig.colorbar(im, ax=ax)
#     xlabel = "Decoder timestep"
#     if info is not None:
#         xlabel += "\n\n" + info
#     plt.xlabel(xlabel)
#     plt.ylabel("Encoder timestep")
#     plt.tight_layout()

#     fig.canvas.draw()
#     data = save_figure_to_numpy(fig)
#     plt.close()
#     return data


# def plot_spectrogram_to_numpy(spectrogram):
#     spectrogram = spectrogram.astype(np.float32)
#     fig, ax = plt.subplots(figsize=(12, 3))
#     im = ax.imshow(
#         spectrogram, aspect="auto", origin="lower", interpolation="none"
#     )
#     plt.colorbar(im, ax=ax)
#     plt.xlabel("Frames")
#     plt.ylabel("Channels")
#     plt.tight_layout()

#     fig.canvas.draw()
#     data = save_figure_to_numpy(fig)
#     plt.close()
#     return data


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


@rank_zero_only
def plots(swriter, imvs, alphas, mel_preds, mel_gts, step, num_plots=4):

    imvs = imvs.detach().cpu().numpy()
    alphas = alphas.detach().cpu().numpy()
    mel_preds = mel_preds.detach().cpu().numpy()
    mel_gts = mel_gts.detach().cpu().numpy()

    # imvs = imvs.double().detach().cpu().numpy()
    # alphas = alphas.double().detach().cpu().numpy()
    # mel_preds = mel_preds.double().detach().cpu().numpy()
    # mel_gts = mel_gts.double().detach().cpu().numpy()
    # logging.info(mel_gts.shape)

    i = 1
    # w, h = plt.figaspect(1.0 / len(imvs))
    # fig = plt.Figure(figsize=(w * 1.3, h * 1.3))

    for imv, alpha, mel_pred, mel_gt in zip(imvs, alphas, mel_preds, mel_gts):
        fig, ax = plt.subplots(4)
        ax[0].plot(range(len(imv)), imv)
        ax[1].imshow(alpha[::-1])
        ax[2].imshow(mel_pred.T)
        ax[3].imshow(mel_gt.T)
        # fig.savefig(f"{output_dir}/step{step}_{i}.png")
        fig.canvas.draw()
        swriter.add_image(
            f"plot_{i}", save_figure_to_numpy(fig), step, dataformats="HWC",
        )
        if step<100:
            log_audio_to_tb(swriter, mel_gt, f"griffin_lim_gt_{i}", step)
        log_audio_to_tb(swriter, mel_pred, f"griffin_lim_pred_{i}", step)
        plt.close()
        i += 1
        if i > num_plots:
            break

