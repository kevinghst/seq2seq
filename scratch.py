
import string
import pdb
import numpy as np

char2int = {
    'c': 10, 's': 11, 't': 12,
    '+': 13, '-': 14, '/': 15, '*': 16, 'p': 17,
    '(': 18, ')': 19,
    'x': 20
}

for i in range(10):
    char2int[str(i)] = i

VOCAB_SIZE = len(char2int)

def translate_str(exp):
    d = {
        'cos': 'COS',
        'sin': 'SIN',
        'tan': 'TAN'
    }
    for fr, to in d.items():
        exp = exp.replace(fr, to)

    # translate all variables to x
    alpha = string.ascii_lowercase
    exp = exp.translate({ord(c):'x' for c in alpha})

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

def str_to_int(str):
    output = []
    for char in str:
        output.append(char2int[char])

    return np.array(output)

def parse(content):
    xs = []
    ys = []

    for eq in content:
        left, right = eq.split("=")
        left = translate_str(left)
        right = translate_str(right)

        left_ints = str_to_int(left)
        right_ints = str_to_int(right) + [21]
        x = np.zeros((len(left_ints), VOCAB_SIZE))

        x[np.arange(len(left_ints)), left_ints] = 1

        xs.append(x)
        ys.append(right_ints)

    return xs, ys

with open('train_small.txt') as f:
    content = f.readlines()


content = [x.strip() for x in content]

xs, ys = parse(content)

exit = "exit"