import argparse
from pathlib import Path

from cosmociety.equilibrium import relax_to_equilibrium
from cosmociety.diagnostics import format_summary, summarize_result
from cosmociety.visualize import plot_equilibrium, plot_transport_diagnostics
from cosmociety.animation import animate_relaxation
from experiments.demo_cases import case_names, get_case


PREVIEW_PARAMS = {
    "n": 100,
    "max_steps": 77_000,
    "save_every": 3_000,
    "tolerance": -1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Cosmociety demo case.")
    parser.add_argument(
        "--case",
        default="baseline_envelope",
        choices=case_names(),
        help="Named demo case to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated plots and animation.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Run a short non-equilibrium preview using the same case parameters. "
            "Outputs are written with a _preview suffix."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second for the generated GIF.",
    )
    parser.add_argument(
        "--convective-transport",
        choices=["excess", "diffusive"],
        default=None,
        help="Override the convective transport law for this run.",
    )
    parser.add_argument(
        "--surface-cooling",
        type=float,
        default=None,
        help="Override the radiative surface cooling strength.",
    )
    parser.add_argument(
        "--convective-max-diffusivity",
        type=float,
        default=None,
        help="Override the maximum convective diffusivity.",
    )
    parser.add_argument(
        "--opacity-temperature-power",
        type=float,
        default=None,
        help="Override temperature dependence in kappa = rho_contrast^q * T^-p.",
    )
    parser.add_argument(
        "--opacity-density-power",
        type=float,
        default=None,
        help="Override density dependence in kappa = rho_contrast^q * T^-p.",
    )
    parser.add_argument(
        "--radiative-density-power",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def run_case(
    case_name: str,
    output_dir: Path | None = None,
    preview: bool = False,
    fps: int | None = None,
    overrides: dict | None = None,
) -> dict:
    case = get_case(case_name)
    output_dir = output_dir or Path("outputs") / "cases" / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    params = dict(case["params"])
    if overrides:
        params.update({key: value for key, value in overrides.items() if value is not None})

    transport = params.get("convective_transport", "excess")
    suffix = f"_{transport}"
    mode = "equilibrium"
    if preview:
        mode = "preview"
        suffix = f"_{transport}_preview"
        params.update(PREVIEW_PARAMS)

    result = relax_to_equilibrium(**params)

    if result["converged_step"] is None and not preview:
        raise RuntimeError("Simulation did not reach equilibrium; skipping diagnostics.")

    plot_equilibrium(result, save_path=str(output_dir / f"radiative_equilibrium{suffix}.png"))
    plot_transport_diagnostics(
        result,
        save_path=str(output_dir / f"transport_diagnostics{suffix}.png"),
    )

    fps = fps if fps is not None else (10 if preview else 30)
    animate_relaxation(
        result,
        save_path=str(output_dir / f"radiative_relaxation{suffix}.gif"),
        fps=fps,
    )

    summary = format_summary(summarize_result(result))
    (output_dir / f"summary{suffix}.txt").write_text(
        f"case: {case_name}\n"
        f"mode: {mode}\n"
        f"description: {case['description']}\n\n"
        f"{summary}\n"
    )

    print(f"Case: {case_name}")
    print(f"Mode: {mode}")
    print(case["description"])
    print("Simulation complete.")
    print(f"Output directory: {output_dir}")
    print(f"Converged step: {result['converged_step']}")
    print(summary)

    return {
        "case": case_name,
        "mode": mode,
        "output_dir": output_dir,
        "suffix": suffix,
        "result": result,
        "summary": summary,
    }


def main():
    args = parse_args()
    output_dir = args.output_dir or Path("outputs") / "cases" / args.case
    overrides = {
        "convective_transport": args.convective_transport,
        "surface_cooling": args.surface_cooling,
        "convective_max_diffusivity": args.convective_max_diffusivity,
        "opacity_temperature_power": args.opacity_temperature_power,
        "opacity_density_power": (
            args.opacity_density_power
            if args.opacity_density_power is not None
            else args.radiative_density_power
        ),
    }
    run_case(
        case_name=args.case,
        output_dir=output_dir,
        preview=args.preview,
        fps=args.fps,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
    
