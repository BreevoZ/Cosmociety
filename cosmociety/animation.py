import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

from .convection import (
    convective_diffusivity_from_gradient,
    convective_diffusivity_from_radiative_demand,
    convective_diffusivity_from_schwarzschild,
    convective_flux_from_adiabatic_excess,
)
from .opacity import radiative_diffusivity_from_state
from .transport import interface_flux


def animate_relaxation(
    result: dict,
    save_path: str = "outputs/radiative_relaxation.gif",
    fps: int = 30,
):
    """
    Animate temperature relaxation toward equilibrium.
    """

    r = result["r"]
    history = result["history"]
    source = result["source"]
    density = result.get("density", np.ones_like(r))
    pressure_density_power = result.get("pressure_density_power", 1.0)
    threshold = result.get("convective_threshold")
    show_convection = bool(result.get("enable_convection", False)) and threshold is not None
    criterion = result.get("convective_criterion", "gradient")
    convective_transport = result.get("convective_transport", "diffusive")
    opacity_temperature_power = result.get(
        "opacity_temperature_power",
        result.get("opacity_power"),
    )
    opacity_density_power = result.get(
        "opacity_density_power",
        result.get("radiative_density_power", 0.0),
    )
    diffusivity_scale = result.get("radiative_diffusivity_scale")
    diffusivity_floor = result.get("radiative_diffusivity_floor", 0.0)
    convective_strength = result.get("convective_strength", 0.05)
    convective_max_diffusivity = result.get("convective_max_diffusivity", 0.05)
    convective_nabla_ad = result.get("convective_nabla_ad", threshold)
    dr = r[1] - r[0]
    r_interface = 0.5 * (r[:-1] + r[1:])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), constrained_layout=True)
    ax_temp, ax_profile, ax_transport, ax_flux = axes.flat

    ymax = max(history.max(), source.max()) * 1.1

    ax_temp.set_xlim(0, 1)
    ax_temp.set_ylim(0, ymax)

    ax_temp.set_xlabel("Normalized radius r/R")
    ax_temp.set_ylabel("Temperature")

    ax_temp.set_title("Temperature")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    convective_spans = []

    # Static source profile (scaled for visibility)
    source_scaled = source / source.max() * ymax * 0.3
    ax_temp.plot(r, source_scaled, "--", label="Core source (scaled)")

    # Dynamic temperature curve
    temp_line, = ax_temp.plot([], [], linewidth=2, label="Temperature T(r)")

    status_text = fig.text(
        0.5,
        0.012,
        "",
        ha="center",
        va="bottom",
        fontsize=9,
        family="monospace",
    )

    legend_handles, legend_labels = ax_temp.get_legend_handles_labels()
    if show_convection:
        legend_handles.append(
            Patch(
                facecolor=(1.0, 0.55, 0.12, 0.22),
                edgecolor="none",
                label="Convective region",
            )
        )
    ax_temp.legend(
        legend_handles,
        legend_labels + (["Convective region"] if show_convection else []),
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
    )

    ax_profile.plot(r, density, label="Density rho")
    ax_profile.plot(r, source / source.max() * density.max(), "--", label="Source scaled")
    ax_profile.set_title("Static profiles")
    ax_profile.set_xlabel("Normalized radius r/R")
    ax_profile.set_ylabel("Profile value")
    ax_profile.legend(loc="upper right", fontsize=7, framealpha=0.85)

    ax_transport.set_title("Transport coefficients")
    ax_transport.set_xlabel("Normalized radius r/R")
    ax_transport.set_ylabel("D")
    ax_transport.set_yscale("log")
    transport_floor = max(diffusivity_floor * 0.5, 1e-12)
    transport_ceiling = max(
        result.get("diffusivity", np.array([1.0])).max(),
        convective_max_diffusivity,
        diffusivity_scale or 1.0,
    ) * 2
    ax_transport.set_ylim(transport_floor, transport_ceiling)
    d_rad_line, = ax_transport.plot([], [], label="D_rad")
    d_conv_line, = ax_transport.plot([], [], label="D_conv")
    d_total_line, = ax_transport.plot([], [], label="D_total")
    ax_transport.legend(loc="lower left", fontsize=7, framealpha=0.85)

    ax_flux.set_title("Flux and luminosity")
    ax_flux.set_xlabel("Normalized radius r/R")
    ax_flux.set_ylabel("Magnitude")
    flux_line, = ax_flux.plot([], [], label="Flux")
    luminosity_line, = ax_flux.plot([], [], label="r^2 flux")
    ax_flux.legend(loc="upper right", fontsize=7, framealpha=0.85)

    def active_regions(active):
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False])
        return zip(starts, ends)

    def radiative_diffusivity(T):
        return radiative_diffusivity_from_state(
            temperature=T,
            density=density,
            diffusivity_scale=diffusivity_scale,
            opacity_temperature_power=opacity_temperature_power,
            opacity_density_power=opacity_density_power,
            diffusivity_floor=diffusivity_floor,
        )

    def convective_state(T, D_rad):
        """Return (cell-centered D_conv, interface active mask) for one frame.

        Dispatches into the same cosmociety.convection criterion functions
        used by equilibrium.relax_to_equilibrium, so this animation cannot
        drift from the physics used to produce the static diagnostics/plots.
        """
        if not show_convection:
            return np.zeros_like(T), np.zeros(len(r) - 1, dtype=bool)

        if criterion == "gradient":
            D_conv, excess = convective_diffusivity_from_gradient(
                temperature=T,
                dr=dr,
                threshold=threshold,
                strength=convective_strength,
                max_diffusivity=convective_max_diffusivity,
            )
        elif criterion == "radiative":
            D_conv, excess = convective_diffusivity_from_radiative_demand(
                source=source,
                radius=r,
                diffusivity=D_rad,
                dr=dr,
                threshold=threshold,
                strength=convective_strength,
                max_diffusivity=convective_max_diffusivity,
            )
        elif criterion == "schwarzschild":
            pressure = density**pressure_density_power * T
            D_conv, excess, _ = convective_diffusivity_from_schwarzschild(
                source=source,
                radius=r,
                temperature=T,
                pressure=pressure,
                diffusivity=D_rad,
                dr=dr,
                nabla_ad=threshold,
                strength=convective_strength,
                max_diffusivity=convective_max_diffusivity,
            )
        else:
            raise ValueError(
                "convective_criterion must be 'gradient', 'radiative', or 'schwarzschild'"
            )

        return D_conv, excess > 0

    def flux_from_components(T, D_rad, D_conv):
        radiative_flux = interface_flux(T, D_rad, dr)
        if convective_transport == "diffusive":
            convective_flux = interface_flux(T, D_conv, dr)
        elif convective_transport == "excess":
            pressure = density**pressure_density_power * T
            convective_flux = convective_flux_from_adiabatic_excess(
                temperature=T,
                pressure=pressure,
                convective_diffusivity=D_conv,
                dr=dr,
                nabla_ad=convective_nabla_ad,
            )
        else:
            raise ValueError("convective_transport must be 'diffusive' or 'excess'")
        flux = radiative_flux + convective_flux
        return flux, r_interface**2 * flux

    flux_min = 0.0
    flux_max = 1.0
    if history.size:
        flux_values = []
        luminosity_values = []
        for T_frame in history:
            D_rad_frame = radiative_diffusivity(T_frame)
            D_conv_frame, _ = convective_state(T_frame, D_rad_frame)
            flux_frame, luminosity_frame = flux_from_components(
                T_frame,
                D_rad_frame,
                D_conv_frame,
            )
            flux_values.append(flux_frame)
            luminosity_values.append(luminosity_frame)

        all_flux = np.concatenate(flux_values + luminosity_values)
        finite_flux = all_flux[np.isfinite(all_flux)]
        if finite_flux.size:
            flux_min = min(float(finite_flux.min()), 0.0)
            flux_max = max(float(finite_flux.max()), 1e-12)
            padding = 0.08 * (flux_max - flux_min)
            flux_min -= padding
            flux_max += padding
    ax_flux.set_ylim(flux_min, flux_max)

    def init():
        temp_line.set_data([], [])
        d_rad_line.set_data([], [])
        d_conv_line.set_data([], [])
        d_total_line.set_data([], [])
        flux_line.set_data([], [])
        luminosity_line.set_data([], [])
        status_text.set_text("")
        return (
            temp_line,
            d_rad_line,
            d_conv_line,
            d_total_line,
            flux_line,
            luminosity_line,
            status_text,
        )

    def update(frame):
        nonlocal convective_spans
        T = history[frame]

        temp_line.set_data(r, T)

        for span in convective_spans:
            span.remove()
        convective_spans = []

        D_rad = radiative_diffusivity(T)
        D_conv, active = convective_state(T, D_rad)
        if show_convection:
            for start, end in active_regions(active):
                convective_spans.append(
                    ax_temp.axvspan(
                        r[start],
                        r[end + 1],
                        color="darkorange",
                        alpha=0.28,
                        linewidth=0,
                        zorder=1,
                    )
                )

        D_total = D_rad + D_conv
        flux, luminosity = flux_from_components(T, D_rad, D_conv)

        d_rad_line.set_data(r, D_rad)
        d_conv_line.set_data(r, np.maximum(D_conv, transport_floor))
        d_total_line.set_data(r, D_total)
        flux_line.set_data(r_interface, flux)
        luminosity_line.set_data(r_interface, luminosity)

        active_fraction = active.mean() if show_convection else 0.0
        status_text.set_text(
            f"frame {frame:03d}/{len(history)-1:03d} | "
            f"conv {active_fraction:6.2%} | "
            f"{criterion}/{convective_transport}"
        )

        return (
            temp_line,
            d_rad_line,
            d_conv_line,
            d_total_line,
            flux_line,
            luminosity_line,
            status_text,
            *convective_spans,
        )

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(history),
        interval=1000 / fps,
        blit=False,
    )

    anim.save(save_path, writer=PillowWriter(fps=fps))

    plt.close(fig)

    print(f"Saved animation to: {save_path}")
