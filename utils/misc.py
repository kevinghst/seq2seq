import librosa
import numpy as np
import editdistance
import torch

EOS_TOKEN = '</s>'

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    print(f"Total Trainable Params: {total_params}")
    return total_params

def check_size(tensor, *args):
    size = [a for a in args]
    assert tensor.size() == torch.Size(size), tensor.size()

def to_mono(y):
    assert y.ndim == 2
    return np.mean(y, axis=1)


def downsample(y, orig_sr, targ_sr):
    if y.dtype != np.float:
        y = y.astype(np.float32)
    return librosa.resample(y, orig_sr=orig_sr, target_sr=targ_sr)


def standardize(x):
    new_x = (x - np.mean(x, 0)) / (np.std(x, 0) + 1e-3)
    return new_x


def edit_distance(guess, truth):
    guess = guess.split(EOS_TOKEN)[0]
    truth = truth[3:].split(EOS_TOKEN)[0]
    return editdistance.eval(guess, truth) / len(truth)


class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__