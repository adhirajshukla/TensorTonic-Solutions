import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(x, dtype=float)              # copy, input untouched

    if rng is not None:
        r = rng.random(x.shape)
    else:
        r = np.random.random(x.shape)

    keep = r >= p                             # drop with probability p
    pattern = keep.astype(float) / (1.0 - p)  # 0 or 1/(1-p)
    output = x * pattern
    return output, pattern