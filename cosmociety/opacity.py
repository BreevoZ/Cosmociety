import numpy as np


def opacity_from_state(
    temperature: np.ndarray,
    density: np.ndarray,
    opacity_temperature_power: float,
    opacity_density_power: float,
) -> np.ndarray:
    """Toy opacity law: kappa = (rho / rho_surface)^q T^-p."""
    density_contrast = density / max(float(density[-1]), 1e-30)
    return density_contrast**opacity_density_power * temperature ** (
        -opacity_temperature_power
    )


def radiative_diffusivity_from_state(
    temperature: np.ndarray,
    density: np.ndarray,
    diffusivity_scale: float,
    opacity_temperature_power: float,
    opacity_density_power: float,
    diffusivity_floor: float,
) -> np.ndarray:
    """Toy radiative diffusivity: D_rad = D0 / kappa."""
    kappa = opacity_from_state(
        temperature=temperature,
        density=density,
        opacity_temperature_power=opacity_temperature_power,
        opacity_density_power=opacity_density_power,
    )
    return np.maximum(diffusivity_scale / kappa, diffusivity_floor)
