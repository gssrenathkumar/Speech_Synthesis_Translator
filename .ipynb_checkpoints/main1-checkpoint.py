import time
import matplotlib
import matplotlib.pylab as plt
import gdown
import IPython.display as ipd
import numpy as np
import torch
import json
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser
import resampy
import scipy.signal

def end_to_end_infer(text, pronounciation_dictionary, show_graphs):
    for i in [x for x in text.split("\n") if len(x)]:
        if not pronounciation_dictionary:
            if i[-1] != ";": i=i+";" 
        else: i = ARPA(i)
        with torch.no_grad(): # save VRAM by not including gradients
            s = np.array(text_to_sequence(i, ['english_cleaners']))[None, :]
            s = torch.autograd.Variable(torch.from_numpy(s)).long()
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(s)
            if show_graphs:
                plot_data((mel_outputs_postnet.float().data.cpu().numpy()[0],
                        alignments.float().data.cpu().numpy()[0].T))
            y_g_hat = hifigan(mel_outputs_postnet.float())
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio_d = denoiser(audio.view(1, -1), strength=35)[:, 0]

            # Resample to 32k
            audio_d = audio_d.cpu().numpy().reshape(-1)

            normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_d))) ** 0.9
            audio_d = audio_d * normalize
            wave = resampy.resample(
                audio_d,
                h.sampling_rate,
                h2.sampling_rate,
                filter="sinc_window",
                window=scipy.signal.windows.hann,
                num_zeros=8,
            )
            wave_out = wave.astype(np.int16)

            # HiFi-GAN super-resolution
            wave = wave / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(torch.device("cpu"))
            new_mel = mel_spectrogram(
                wave.unsqueeze(0),
                h2.n_fft,
                h2.num_mels,
                h2.sampling_rate,
                h2.hop_size,
                h2.win_size,
                h2.fmin,
                h2.fmax,
            )
            y_g_hat2 = hifigan_sr(new_mel)
            audio2 = y_g_hat2.squeeze()
            audio2 = audio2 * MAX_WAV_VALUE
            audio_d2 = denoiser(audio2.view(1, -1), strength=35)[:, 0]

            # High-pass filter, mixing and denormalizing
            audio_d2 = audio_d2.cpu().numpy().reshape(-1)
            b = scipy.signal.firwin(
                101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False
            )
            y = scipy.signal.lfilter(b, [1.0], audio_d2)
            y *= superres_strength
            y_out = y.astype(np.int16)
            y_padded = np.zeros(wave_out.shape)
            y_padded[: y_out.shape[0]] = y_out
            sr_mix = wave_out + y_padded
            sr_mix = sr_mix / normalize


if previous_tt2_id != TACOTRON2_ID:
    print("Updating Models")
    model, hparams = get_Tactron2(TACOTRON2_ID)
    hifigan, h, denoiser = get_hifigan(HIFIGAN_ID, "config_v1")
    previous_tt2_id = TACOTRON2_ID

pronounciation_dictionary = True 

show_graphs = True 
max_duration =  30
model.decoder.max_decoder_steps = max_duration * 80
stop_threshold = 0.8
model.decoder.gate_threshold = stop_threshold
superres_strength = 6.0

print(f"Current Config:\npronounciation_dictionary: {pronounciation_dictionary}\nshow_graphs: {show_graphs}\nmax_duration (in seconds): {max_duration}\nstop_threshold: {stop_threshold}\nsuperres_strength: {superres_strength}\n\n")

time.sleep(1)
print("Enter/Paste your text.")
contents = []
print("-"*50)
for i in translated_text:
  line = i
  end_to_end_infer(line, not pronounciation_dictionary, show_graphs)