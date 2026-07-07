import numpy as np


def convective_diffusivity_from_gradient(
    temperature: np.ndarray,
    dr: float,
    threshold: float,
    strength: float,
    max_diffusivity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a toy convective diffusivity triggered by steep temperature gradients.

    Convection appears where the outward temperature drop is steeper than
    threshold. The returned cell-centered diffusivity can be added to the
    radiative diffusivity.
    """
    outward_drop = -np.diff(temperature) / dr
    excess = np.maximum(outward_drop - threshold, 0.0)
    threshold_scale = max(threshold, 1e-30)
    interface_diffusivity = np.minimum(
        strength * excess / threshold_scale,
        max_diffusivity,
    )

    cell_diffusivity = np.zeros_like(temperature)
    cell_diffusivity[0] = interface_diffusivity[0]
    cell_diffusivity[-1] = interface_diffusivity[-1]
    cell_diffusivity[1:-1] = 0.5 * (
        interface_diffusivity[:-1] + interface_diffusivity[1:]
    )

    return cell_diffusivity, excess


def convective_diffusivity_from_radiative_demand(
    source: np.ndarray,
    radius: np.ndarray,
    diffusivity: np.ndarray,
    dr: float,
    threshold: float,
    strength: float,
    max_diffusivity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return convective diffusivity where radiation alone needs a steep gradient.

    This is a toy Schwarzschild-like trigger. It estimates the luminosity that
    must pass through each interface from the enclosed source, then asks what
    temperature gradient radiation alone would need to carry that luminosity.
    """
    r_interface = 0.5 * (radius[:-1] + radius[1:])
    source_shells = source * radius**2 * dr
    enclosed_luminosity = np.cumsum(source_shells)[:-1]

    diffusivity_interface = 0.5 * (diffusivity[:-1] + diffusivity[1:])
    radiative_gradient = enclosed_luminosity / (
        np.maximum(r_interface**2 * diffusivity_interface, 1e-30)
    )
    excess = np.maximum(radiative_gradient - threshold, 0.0)
    threshold_scale = max(threshold, 1e-30)
    interface_diffusivity = np.minimum(
        strength * excess / threshold_scale,
        max_diffusivity,
    )

    cell_diffusivity = np.zeros_like(source)
    cell_diffusivity[0] = interface_diffusivity[0]
    cell_diffusivity[-1] = interface_diffusivity[-1]
    cell_diffusivity[1:-1] = 0.5 * (
        interface_diffusivity[:-1] + interface_diffusivity[1:]
    )

    return cell_diffusivity, excess


def convective_diffusivity_from_schwarzschild(
    source: np.ndarray,
    radius: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    diffusivity: np.ndarray,
    dr: float,
    nabla_ad: float,
    strength: float,
    max_diffusivity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return convective diffusivity from a toy Schwarzschild criterion.

    It estimates the temperature gradient radiation alone would need to carry
    the enclosed luminosity, converts that to nabla_rad = d ln T / d ln P, and
    triggers convection where nabla_rad > nabla_ad.
    """
    r_interface = 0.5 * (radius[:-1] + radius[1:])
    source_shells = source * radius**2 * dr
    enclosed_luminosity = np.cumsum(source_shells)[:-1]

    diffusivity_interface = 0.5 * (diffusivity[:-1] + diffusivity[1:])
    required_temperature_drop = enclosed_luminosity / (
        np.maximum(r_interface**2 * diffusivity_interface, 1e-30)
    )

    temperature_interface = 0.5 * (temperature[:-1] + temperature[1:])
    pressure_interface = 0.5 * (pressure[:-1] + pressure[1:])
    pressure_drop = -np.diff(pressure) / dr

    nabla_rad = (
        pressure_interface
        * required_temperature_drop
        / np.maximum(temperature_interface * pressure_drop, 1e-30)
    )
    excess = np.maximum(nabla_rad - nabla_ad, 0.0)
    interface_diffusivity = np.minimum(
        strength * excess / max(nabla_ad, 1e-30),
        max_diffusivity,
    )

    cell_diffusivity = np.zeros_like(source)
    cell_diffusivity[0] = interface_diffusivity[0]
    cell_diffusivity[-1] = interface_diffusivity[-1]
    cell_diffusivity[1:-1] = 0.5 * (
        interface_diffusivity[:-1] + interface_diffusivity[1:]
    )

    return cell_diffusivity, excess, nabla_rad


def convective_flux_from_adiabatic_excess(
    temperature: np.ndarray,
    pressure: np.ndarray,
    convective_diffusivity: np.ndarray,
    dr: float,
    nabla_ad: float,
) -> np.ndarray:
    """
    Return convective flux that only transports superadiabatic temperature drop.

    The target retained gradient is the toy adiabatic gradient:

        dT/dr_ad = nabla_ad * T/P * dP/dr

    Since outward heat flux is positive, this uses outward temperature drops.
    """
    outward_temperature_drop = -np.diff(temperature) / dr
    outward_pressure_drop = np.maximum(-np.diff(pressure) / dr, 0.0)
    temperature_interface = 0.5 * (temperature[:-1] + temperature[1:])
    pressure_interface = 0.5 * (pressure[:-1] + pressure[1:])
    diffusivity_interface = 0.5 * (
        convective_diffusivity[:-1] + convective_diffusivity[1:]
    )

    adiabatic_temperature_drop = (
        nabla_ad
        * temperature_interface
        * outward_pressure_drop
        / np.maximum(pressure_interface, 1e-30)
    )
    superadiabatic_drop = np.maximum(
        outward_temperature_drop - adiabatic_temperature_drop,
        0.0,
    )
    return diffusivity_interface * superadiabatic_drop
