import argparse
import csv
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cosmociety.diagnostics import summarize_result
from cosmociety.equilibrium import relax_to_equilibrium


OUTPUT_PATH = Path("outputs/convection_scan.csv")


SCAN = {
    "opacity_temperature_power": [8.0, 10.0, 12.0],
    "convective_max_diffusivity": [5e-4, 1e-3, 2e-3],
    "convective_nabla_ad": [0.35, 0.4, 0.45],
}


BASE_PARAMS = {
    "n": 120,
    "max_steps": 350_000,
    "tolerance": 1.5e-7,
    "save_every": 30_000,
}

QUICK_SCAN = {
    "opacity_temperature_power": [8.0, 10.0],
    "convective_max_diffusivity": [5e-4, 1e-3],
    "convective_nabla_ad": [0.4],
}


QUICK_PARAMS = {
    "n": 100,
    "max_steps": 120_000,
    "tolerance": 2e-7,
    "save_every": 20_000,
}


FIELDS = [
    "run",
    "opacity_temperature_power",
    "opacity_density_power",
    "convective_max_diffusivity",
    "convective_nabla_ad",
    "convective_transport",
    "opacity_power",
    "radiative_density_power",
    "converged",
    "converged_step",
    "regime",
    "structural_regime",
    "final_delta",
    "temperature_contrast",
    "temperature_center",
    "temperature_surface",
    "convective_fraction",
    "num_regions",
    "convective_regions",
    "convective_inner_radius",
    "convective_outer_radius",
    "convective_region_count",
    "has_convective_core",
    "has_convective_envelope",
    "max_radiative_diffusivity",
    "max_convective_diffusivity",
    "max_total_diffusivity",
    "max_luminosity",
    "max_nabla_rad",
]


def parameter_sets(scan):
    keys = list(scan)
    for values in product(*(scan[key] for key in keys)):
        yield dict(zip(keys, values))


def print_row(row):
    status = "ok" if row["converged"] else "open"
    region = row["convective_regions"]

    print(
        f"{row['run']:02d} {status:4s} "
        f"p={row['opacity_temperature_power']:<4.1f} "
        f"Dcmax={row['convective_max_diffusivity']:<7.1e} "
        f"nad={row['convective_nabla_ad']:<4.2f} "
        f"regime={row['regime']:<24s} "
        f"conv={row['convective_fraction']:<5.2f} "
        f"regions={row['convective_region_count']}:{region:<18s} "
        f"Tc/Ts={row['temperature_contrast']:.2f} "
        f"delta={row['final_delta']:.1e}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Scan toy convection parameters.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small fast scan for smoke testing.",
    )
    args = parser.parse_args()

    scan = QUICK_SCAN if args.quick else SCAN
    base_params = QUICK_PARAMS if args.quick else BASE_PARAMS
    output_path = (
        OUTPUT_PATH.with_name("convection_scan_quick.csv")
        if args.quick
        else OUTPUT_PATH
    )

    rows = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for run, params in enumerate(parameter_sets(scan), start=1):
        result = relax_to_equilibrium(**base_params, **params)
        row = {"run": run, **params, **summarize_result(result)}
        rows.append(row)
        print_row(row)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})

    print(f"\nSaved scan to {output_path}")


if __name__ == "__main__":
    main()
