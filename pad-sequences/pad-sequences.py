import numpy as np

def pad_sequences(sequences, pad_value=0, max_len=None):
    if len(sequences) == 0:
        return np.empty((0, 0), dtype=int)

    target = max_len if max_len is not None else max(len(s) for s in sequences)

    out = np.full((len(sequences), target), pad_value, dtype=int)
    for i, seq in enumerate(sequences):
        n = min(len(seq), target)
        out[i, :n] = seq[:n]
    return out