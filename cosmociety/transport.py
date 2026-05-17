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
    space_temperature: float,
    surface_cooling: float,
    dr: float,
    radius: np.ndarray | None = None,
    geometry: str = "spherical",
) -> np.ndarray:
    """
    One explicit flux-form diffusion step with radiative surface cooling.

    Solves the toy equation in either cartesian or spherical form:

        cartesian: dT/dt = d/dr [ D(r) dT/dr ] + source
        spherical: dT/dt = 1/r^2 d/dr [ r^2 D(r) dT/dr ] + source

    Surface is not fixed anymore.
    Instead, the outer boundary cools radiatively:

        cooling ~ T_surface^4 - T_space^4

    This is a boundary flux, so it is divided by dr when applied to the
    outermost cell.
    """
    T = temperature.copy()

    # Heat flux across interfaces
    flux = interface_flux(T, diffusivity, dr)

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
    T += dt * (dT + source)

    # Center symmetry
    T[0] = T[1]

    # Radiative cooling at surface
    cooling = surface_cooling * (T[-1]**4 - space_temperature**4)
    T[-1] -= dt * cooling / dr

    # Numerical safety
    T = np.maximum(T, space_temperature)

    return T
