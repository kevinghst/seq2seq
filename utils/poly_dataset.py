import numpy as np
import torch

import string
from torch.utils import data
from random import choice, randrange
from itertools import zip_longest


import pdb

def batch(iterable, n=1):
    args = [iter(iterable)] * n
    return zip_longest(*args)


def pad_tensor(vec, pad, value=0, dim=0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = pad - vec.shape[0]

    if len(vec.shape) == 2:
        zeros = torch.ones((pad_size, vec.shape[-1])) * value
    elif len(vec.shape) == 1:
        zeros = torch.ones((pad_size,)) * value
    else:
        raise NotImplementedError
    return torch.cat([torch.Tensor(vec), zeros], dim=dim)


def pad_collate(batch, values=(0, 0), dim=0):
    """
    args:
        batch - list of (tensor, label)

    reutrn:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
        ws - a tensor of sequence lengths
    """

    sequence_lengths = torch.Tensor([int(x[0].shape[dim]) for x in batch])
    sequence_lengths, xids = sequence_lengths.sort(descending=True)
    target_lengths = torch.Tensor([int(x[1].shape[dim]) for x in batch])
    target_lengths, yids = target_lengths.sort(descending=True)
    # find longest sequence
    src_max_len = max(map(lambda x: x[0].shape[dim], batch))
    tgt_max_len = max(map(lambda x: x[1].shape[dim], batch))
    # pad according to max_len
    batch = [(pad_tensor(x, pad=src_max_len, dim=dim), pad_tensor(y, pad=tgt_max_len, dim=dim)) for (x, y) in batch]

    # stack all
    xs = torch.stack([x[0] for x in batch], dim=0)
    ys = torch.stack([x[1] for x in batch]).int()
    xs = xs[xids]
    ys = ys[yids]
    return xs, ys, sequence_lengths.int(), target_lengths.int()


class PolyDataset(data.Dataset):
    """
    https://talbaumel.github.io/blog/attention/
    """
    def __init__(self, filepath='train_small.txt'):
        self.SOS = "<s>"  # all strings will end with the End Of String token
        self.EOS = "</s>"  # all strings will end with the End Of String token

        self.char2int = {
            'c': 10, 's': 11, 't': 12,
            '+': 13, '-': 14, '/': 15, '*': 16, 'p': 17,
            '(': 18, ')': 19,
            'x': 20
        }

        for i in range(10):
            self.char2int[str(i)] = i

        self.VOCAB_SIZE = len(self.char2int)

        with open(filepath) as f:
            content = f.readlines()

        content = [x.strip() for x in content]

        self.set = self.parse(content)

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):
        return self.set[item]

    def translate_str(self, exp):
        d = {
            'cos': 'COS',
            'sin': 'SIN',
            'tan': 'TAN'
        }
        for fr, to in d.items():
            exp = exp.replace(fr, to)

        # translate all variables to x
        alpha = string.ascii_lowercase
        exp = exp.translate({ord(c): 'x' for c in alpha})

        # translate expressions to single character variables
        d = {
            '**': 'p',
            'COS(x)': 'c',
            'SIN(x)': 's',
            'TAN(x)': 't'
        }

        for fr, to in d.items():
            exp = exp.replace(fr, to)

        return exp.replace(" ", "")

    def str_to_int(self, str):
        output = []
        for char in str:
            output.append(self.char2int[char])
        return output

    def parse(self, content):
        pairs = []

        for eq in content:
            left, right = eq.split("=")
            left = self.translate_str(left)
            right = self.translate_str(right)

            left_ints = self.str_to_int(left)
            right_ints = self.str_to_int(right) + [21]
            x = np.zeros((len(left_ints), self.VOCAB_SIZE))

            x[np.arange(len(left_ints)), left_ints] = 1

            pairs.append((x, np.array(right_ints)))

        return pairs


