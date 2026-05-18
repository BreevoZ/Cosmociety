import numpy as np


def core_source(r: np.ndarray, width: float = 0.12, strength: float = 1.0) -> np.ndarray:
    """Simple nuclear heating source concentrated near the core."""
    return strength * np.exp(-(r / width) ** 2)


def initial_temperature(r: np.ndarray, surface_temperature: float = 0.1) -> np.ndarray:
    """Smooth initial temperature guess: hot center, cool surface."""
    return surface_temperature + (1.0 - surface_temperature) * (1.0 - r**2)


def density_profile(
    r: np.ndarray,
    surface_density: float = 0.05,
    central_density: float = 1.0,
    concentration: float = 2.0,
) -> np.ndarray:
    """Toy stellar density profile: dense core, light outer envelope."""
    if surface_density <= 0:
        raise ValueError("surface_density must be positive")
    if central_density < surface_density:
        raise ValueError("central_density must be at least surface_density")

    envelope = np.maximum(1.0 - r**2, 0.0) ** concentration
    return surface_density + (central_density - surface_density) * envelope
