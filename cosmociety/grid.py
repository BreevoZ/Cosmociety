import numpy as np


def radial_grid(n: int = 200) -> np.ndarray:
    """Return normalized radius grid from center r=0 to surface r=1."""
    if n < 3:
        raise ValueError("n must be at least 3")
    return np.linspace(0.0, 1.0, n)
