import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    pos = np.arange(seq_len)[:, None]                       # (seq_len, 1)
    i = np.arange(0, d_model, 2)[None, :]                   # (1, ceil(d_model/2))
    angles = pos / np.power(base, i / d_model)              # (seq_len, ceil(d_model/2))

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)                            # even columns
    pe[:, 1::2] = np.cos(angles[:, : d_model // 2])         # odd columns
    return pe