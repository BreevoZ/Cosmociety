import numpy as np


def convective_regions(radius: np.ndarray, active: np.ndarray) -> list[tuple[float, float]]:
    """Return contiguous convective radial regions from a cell-centered mask."""
    if not np.any(active):
        return []

    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    return [(float(radius[start]), float(radius[end])) for start, end in zip(starts, ends)]


def format_regions(regions: list[tuple[float, float]]) -> str:
    """Return compact region text such as '0.000-0.080;0.540-1.000'."""
    if not regions:
        return "none"
    return ";".join(f"{inner:.3f}-{outer:.3f}" for inner, outer in regions)


def summarize_result(result: dict) -> dict:
    """Return scalar diagnostics for comparing toy stellar model runs."""
    r = result["r"]
    temperature = result["temperature"]
    convective_diffusivity = result.get("convective_diffusivity", np.zeros_like(r))
    active = convective_diffusivity > 0
    active_r = r[active]
    regions = convective_regions(r, active)
    deltas = result.get("deltas", np.array([]))
    timesteps = result.get("timesteps", np.array([]))

    summary = {
        "converged": result.get("converged_step") is not None,
        "converged_step": result.get("converged_step"),
        "final_delta": float(deltas[-1]) if len(deltas) else np.nan,
        "min_timestep": float(np.min(timesteps)) if len(timesteps) else np.nan,
        "max_timestep": float(np.max(timesteps)) if len(timesteps) else np.nan,
        "temperature_center": float(temperature[0]),
        "temperature_surface": float(temperature[-1]),
        "temperature_contrast": float(temperature[0] / temperature[-1]),
        "temperature_max": float(np.max(temperature)),
        "temperature_min": float(np.min(temperature)),
        "convective_fraction": float(np.mean(active)),
        "convective_inner_radius": float(active_r[0]) if active_r.size else np.nan,
        "convective_outer_radius": float(active_r[-1]) if active_r.size else np.nan,
        "num_regions": len(regions),
        "convective_region_count": len(regions),
        "convective_regions": format_regions(regions),
        "has_convective_core": bool(regions and regions[0][0] <= r[1]),
        "has_convective_envelope": bool(regions and regions[-1][1] >= r[-2]),
        "max_radiative_diffusivity": float(np.max(result["radiative_diffusivity"])),
        "max_convective_diffusivity": float(np.max(convective_diffusivity)),
        "max_total_diffusivity": float(np.max(result["diffusivity"])),
        "max_luminosity": float(np.max(result["luminosity"])),
        "geometry": result.get("geometry"),
        "convective_criterion": result.get("convective_criterion"),
        "opacity_power": result.get("opacity_power"),
        "surface_density": float(result["density"][-1]) if "density" in result else np.nan,
        "central_density": float(result["density"][0]) if "density" in result else np.nan,
    }

    if "nabla_rad" in result:
        summary["max_nabla_rad"] = float(np.max(result["nabla_rad"]))

    return summary


def format_summary(summary: dict) -> str:
    """Format the most important diagnostics as a compact human-readable block."""
    convective_inner = summary["convective_inner_radius"]
    convective_region = summary.get("convective_regions", "none")

    converged_step = summary["converged_step"]
    if converged_step is None:
        converged = "no"
    else:
        converged = f"yes, step {converged_step}"

    return "\n".join(
        [
            f"converged: {converged}",
            f"final_delta: {summary['final_delta']:.3e}",
            f"T_center/T_surface: {summary['temperature_contrast']:.3f}",
            f"T_center: {summary['temperature_center']:.3f}",
            f"T_surface: {summary['temperature_surface']:.3f}",
            f"convective_fraction: {summary['convective_fraction']:.3f}",
            f"convective_regions: {convective_region}",
            f"convective_region_count: {summary['convective_region_count']}",
            f"max_D_rad: {summary['max_radiative_diffusivity']:.3e}",
            f"max_D_conv: {summary['max_convective_diffusivity']:.3e}",
            f"max_luminosity: {summary['max_luminosity']:.3e}",
        ]
    )
