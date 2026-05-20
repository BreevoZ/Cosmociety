import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.demo_cases import case_names
from main import run_case


def parse_args():
    parser = argparse.ArgumentParser(description="Run all Cosmociety demo cases.")
    parser.add_argument(
        "--equilibrium",
        action="store_true",
        help="Run each case to equilibrium. Default is quick preview mode.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=case_names(),
        help="Run only this case. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "cases",
        help="Root directory for per-case outputs.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second for generated GIFs.",
    )
    parser.add_argument(
        "--convective-transport",
        choices=["excess", "diffusive"],
        default=None,
        help="Override the convective transport law for every case.",
    )
    parser.add_argument(
        "--surface-cooling",
        type=float,
        default=None,
        help="Override the radiative surface cooling strength for every case.",
    )
    parser.add_argument(
        "--convective-max-diffusivity",
        type=float,
        default=None,
        help="Override the maximum convective diffusivity for every case.",
    )
    parser.add_argument(
        "--radiative-density-power",
        type=float,
        default=None,
        help="Override density dependence in D_rad for every case.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_cases = args.case or case_names()
    preview = not args.equilibrium
    overrides = {
        "convective_transport": args.convective_transport,
        "surface_cooling": args.surface_cooling,
        "convective_max_diffusivity": args.convective_max_diffusivity,
        "radiative_density_power": args.radiative_density_power,
    }

    print(f"Running {len(selected_cases)} case(s) in {'preview' if preview else 'equilibrium'} mode.")
    for index, case_name in enumerate(selected_cases, start=1):
        print(f"\n[{index}/{len(selected_cases)}] {case_name}")
        run_case(
            case_name=case_name,
            output_dir=args.output_root / case_name,
            preview=preview,
            fps=args.fps,
            overrides=overrides,
        )

    print(f"\nFinished. Outputs are under: {args.output_root}")


if __name__ == "__main__":
    main()
