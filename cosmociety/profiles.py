import numpy as np


def core_source(r: np.ndarray, width: float = 0.12, strength: float = 1.0) -> np.ndarray:
    """Simple nuclear heating source concentrated near the core."""
    return strength * np.exp(-(r / width) ** 2)


def initial_temperature(r: np.ndarray, surface_temperature: float = 0.1) -> np.ndarray:
    """Smooth initial temperature guess: hot center, cool surface."""
    return surface_temperature + (1.0 - surface_temperature) * (1.0 - r**2)
