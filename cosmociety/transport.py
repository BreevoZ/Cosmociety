import numpy as np


def interface_flux(
    temperature: np.ndarray,
    diffusivity: float | np.ndarray,
    dr: float,
) -> np.ndarray:
    """
    Return heat flux across interfaces between radial cells.

    Positive flux means heat flows outward.
    """
    T = np.asarray(temperature, dtype=float)

    if np.isscalar(diffusivity):
        D = np.full_like(T, float(diffusivity))
    else:
        D = np.asarray(diffusivity, dtype=float)

    if D.shape != T.shape:
        raise ValueError("diffusivity must be a scalar or have the same shape as temperature")

    D_interface = 0.5 * (D[:-1] + D[1:])
    return -D_interface * np.diff(T) / dr


def diffusion_step(
    temperature: np.ndarray,
    source: np.ndarray,
    dt: float,
    diffusivity: float | np.ndarray,
    heat_capacity: float | np.ndarray,
    space_temperature: float,
    surface_cooling: float,
    dr: float,
    radius: np.ndarray | None = None,
    geometry: str = "spherical",
    extra_interface_flux: np.ndarray | None = None,
) -> np.ndarray:
    """
    One explicit flux-form diffusion step with radiative surface cooling.

    Solves the toy equation in either cartesian or spherical form:

        C(r) dT/dt = div[ D(r) grad(T) ] + source

    Surface is not fixed anymore.
    Instead, the outer boundary cools radiatively:

        cooling ~ T_surface^4 - T_space^4

    This is a boundary flux, so it is divided by dr when applied to the
    outermost cell.
    """
    T = temperature.copy()
    if np.isscalar(heat_capacity):
        C = np.full_like(T, float(heat_capacity))
    else:
        C = np.asarray(heat_capacity, dtype=float)

    if C.shape != T.shape:
        raise ValueError("heat_capacity must be a scalar or have the same shape as temperature")
    if np.any(C <= 0):
        raise ValueError("heat_capacity must be positive everywhere")

    # Heat flux across interfaces
    flux = interface_flux(T, diffusivity, dr)
    if extra_interface_flux is not None:
        extra_flux = np.asarray(extra_interface_flux, dtype=float)
        if extra_flux.shape != flux.shape:
            raise ValueError("extra_interface_flux must have one value per interface")
        flux = flux + extra_flux

    # Flux divergence
    dT = np.zeros_like(T)
    if geometry == "cartesian":
        dT[1:-1] = -(flux[1:] - flux[:-1]) / dr
        dT[-1] = flux[-1] / dr
    elif geometry == "spherical":
        if radius is None:
            raise ValueError("radius is required for spherical diffusion")

        r = np.asarray(radius, dtype=float)
        if r.shape != T.shape:
            raise ValueError("radius must have the same shape as temperature")

        r_interface = 0.5 * (r[:-1] + r[1:])
        luminosity = r_interface**2 * flux

        dT[1:-1] = -(luminosity[1:] - luminosity[:-1]) / (r[1:-1] ** 2 * dr)
        dT[-1] = luminosity[-1] / (r[-1] ** 2 * dr)
    else:
        raise ValueError("geometry must be 'cartesian' or 'spherical'")

    # Heating
    T += dt * (dT + source) / C

    # Center symmetry
    T[0] = T[1]

    # Radiative cooling at surface
    cooling = surface_cooling * (T[-1]**4 - space_temperature**4)
    T[-1] -= dt * cooling / (C[-1] * dr)

    # Numerical safety
    T = np.maximum(T, space_temperature)

    return T
