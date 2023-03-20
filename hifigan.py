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


graph_width = 900
graph_height = 360


def plot_data(data, figsize=(int(graph_width/100), int(graph_height/100))):
    %matplotlib inline
    """Function to plot wav audio data and the interpretation"""
    f, a = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        a[i].imshow(data[i], aspect='auto', origin='bottom', 
                    interpolation='none', cmap='inferno')
    f.canvas.draw()
    plt.show()

    
    
def get_hifigan(MODEL_ID, conf_name):
    # Download HiFi-GAN
    pretrained_model = 'hifimodel_' + conf_name
    #gdown.download(d+MODEL_ID, pretrained_model, quiet=False)

    if MODEL_ID == 1:
      !wget "https://github.com/justinjohn0306/tacotron2/releases/download/assets/Superres_Twilight_33000" -O $pretrained_model
    elif MODEL_ID == "universal":
      !wget "https://github.com/justinjohn0306/tacotron2/releases/download/assets/g_02500000" -O $pretrained_model
    else:
      gdown.download(d+MODEL_ID, pretrained_model, quiet=False)

    if not exists(pretrained_model):
        raise Exception("HiFI-GAN model failed to download!")

    # Load HiFi-GAN
    conf = os.path.join("hifi-gan", conf_name + ".json")
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device("cpu"))
    state_dict_g = torch.load(pretrained_model, map_location=torch.device("cpu"))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="normal")
    return hifigan, h, denoiser

# Download character HiFi-GAN
hifigan, h, denoiser = get_hifigan(HIFIGAN_ID, "config_v1")
# Download super-resolution HiFi-GAN
hifigan_sr, h2, denoiser_sr = get_hifigan(1, "config_32k")