import numpy as np


def diffusion_step(
    temperature: np.ndarray,
    source: np.ndarray,
    dt: float,
    diffusivity: float,
    surface_temperature: float,
    dr: float,
) -> np.ndarray:
    T = temperature.copy()

    laplacian = np.zeros_like(T)
    laplacian[1:-1] = (T[:-2] - 2.0 * T[1:-1] + T[2:]) / dr**2

    T += dt * (diffusivity * laplacian + source)

    T[0] = T[1]
    T[-1] = surface_temperature

    return T