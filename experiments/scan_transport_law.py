import argparse
import csv
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cosmociety.diagnostics import summarize_result
from cosmociety.equilibrium import relax_to_equilibrium


OUTPUT_PATH = Path("outputs/transport_law_scan.csv")


SCAN = {
    "convective_transport": ["excess", "diffusive"],
    "surface_cooling": [3.0, 10.0, 30.0],
    "convective_max_diffusivity": [5e-4, 1e-3, 3e-3, 1e-2],
}


BASE_PARAMS = {
    "n": 100,
    "max_steps": 180_000,
    "tolerance": 2e-7,
    "save_every": 30_000,
    "opacity_power": 10.0,
    "convective_nabla_ad": 0.4,
}


FIELDS = [
    "run",
    "convective_transport",
    "surface_cooling",
    "convective_max_diffusivity",
    "converged",
    "converged_step",
    "final_delta",
    "temperature_contrast",
    "outer_temperature_ratio",
    "outer_temperature_drop",
    "temperature_center",
    "temperature_surface",
    "convective_fraction",
    "convective_regions",
    "convective_region_count",
    "has_convective_core",
    "has_convective_envelope",
    "max_radiative_diffusivity",
    "max_convective_diffusivity",
    "max_luminosity",
    "max_nabla_rad",
]


def parameter_sets(scan):
    keys = list(scan)
    for values in product(*(scan[key] for key in keys)):
        yield dict(zip(keys, values))


def outer_surface_metrics(result):
    r = result["r"]
    temperature = result["temperature"]
    outer_index = int(np.searchsorted(r, 0.9))
    outer_temperature = temperature[outer_index]
    surface_temperature = temperature[-1]
    return {
        "outer_temperature_ratio": float(outer_temperature / surface_temperature),
        "outer_temperature_drop": float(outer_temperature - surface_temperature),
    }


def print_row(row):
    status = "ok" if row["converged"] else "open"
    print(
        f"{row['run']:02d} {status:4s} "
        f"{row['convective_transport']:<9s} "
        f"cool={row['surface_cooling']:<4.0f} "
        f"Dcmax={row['convective_max_diffusivity']:<7.1e} "
        f"Tc/Ts={row['temperature_contrast']:<6.2f} "
        f"T90/Ts={row['outer_temperature_ratio']:<5.2f} "
        f"conv={row['convective_fraction']:<5.2f} "
        f"regions={row['convective_regions']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare diffusive and superadiabatic-excess convection transport."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="CSV output path.",
    )
    args = parser.parse_args()

    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for run, params in enumerate(parameter_sets(SCAN), start=1):
        result = relax_to_equilibrium(**BASE_PARAMS, **params)
        row = {
            "run": run,
            **params,
            **summarize_result(result),
            **outer_surface_metrics(result),
        }
        rows.append(row)
        print_row(row)

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})

    print(f"\nSaved scan to {args.output}")


if __name__ == "__main__":
    main()
