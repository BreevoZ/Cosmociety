import argparse
from pathlib import Path
from time import perf_counter

from cosmociety.artifacts import export_result, resolved_parameters
from cosmociety.equilibrium import relax_to_equilibrium
from cosmociety.diagnostics import format_summary, summarize_result
from cosmociety.visualize import plot_equilibrium, plot_transport_diagnostics
from cosmociety.animation import animate_relaxation
from cosmociety.cases import case_names, get_case


PREVIEW_PARAMS = {
    "n": 100,
    "max_steps": 77_700,
    "save_every": 3_000,
    "tolerance": -1.0,
}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def grid_size(value: str) -> int:
    number = positive_int(value)
    if number < 3:
        raise argparse.ArgumentTypeError("must be at least 3")
    return number


def parse_args(argv=None):
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
        type=positive_int,
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
    parser.add_argument("--no-animation", action="store_true", help="Skip GIF rendering.")
    parser.add_argument("--n", type=grid_size, default=None, help="Number of radial grid points.")
    parser.add_argument("--max-steps", type=positive_int, default=None, help="Maximum solver steps.")
    parser.add_argument("--save-every", type=positive_int, default=None, help="Snapshot interval in steps.")
    return parser.parse_args(argv)


def run_case(
    case_name: str,
    output_dir: Path | None = None,
    preview: bool = False,
    fps: int | None = None,
    overrides: dict | None = None,
    animation: bool = True,
) -> dict:
    case = get_case(case_name)
    output_dir = output_dir or Path("outputs") / "cases" / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    params = dict(case["params"])
    mode = "equilibrium"
    if preview:
        mode = "preview"
        params.update(PREVIEW_PARAMS)
    if overrides:
        params.update({key: value for key, value in overrides.items() if value is not None})
    params = resolved_parameters(params)
    transport = params["convective_transport"]
    suffix = f"_{transport}" + ("_preview" if preview else "")

    print(f"Running {case_name} ({mode}, n={params['n']}, max_steps={params['max_steps']})...", flush=True)
    started = perf_counter()
    result = relax_to_equilibrium(**params)
    elapsed_seconds = perf_counter() - started

    diagnostics = summarize_result(result)
    export_result(
        result, params, diagnostics, output_dir, suffix, case_name, mode, elapsed_seconds,
    )

    temperature_path = output_dir / f"radiative_equilibrium{suffix}.png"
    transport_path = output_dir / f"transport_diagnostics{suffix}.png"
    plot_equilibrium(result, save_path=str(temperature_path))
    plot_transport_diagnostics(
        result,
        save_path=str(transport_path),
    )

    fps = fps if fps is not None else (10 if preview else 30)
    if animation:
        animation_path = output_dir / f"radiative_relaxation{suffix}.gif"
        animate_relaxation(result, save_path=str(animation_path), fps=fps)

    summary = format_summary(diagnostics)
    summary_path = output_dir / f"summary{suffix}.txt"
    summary_path.write_text(
        f"case: {case_name}\n"
        f"mode: {mode}\n"
        f"description: {case['description']}\n\n"
        f"{summary}\n",
        encoding="utf-8",
    )

    print(f"Case: {case_name}")
    print(f"Mode: {mode}")
    print(case["description"])
    print("Simulation complete.")
    print(f"Output directory: {output_dir}")
    print(f"Converged step: {result['converged_step']}")
    print(summary)
    if result["converged_step"] is None and not preview:
        print("Requested tolerance was not reached. Saved the final state with an open_ regime.")

    return {
        "case": case_name,
        "mode": mode,
        "output_dir": output_dir,
        "suffix": suffix,
        "result": result,
        "summary": summary,
        "diagnostics": diagnostics,
        "parameters": params,
        "elapsed_seconds": elapsed_seconds,
    }


def main(argv=None):
    args = parse_args(argv)
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
        "n": args.n,
        "max_steps": args.max_steps,
        "save_every": args.save_every,
    }
    run = run_case(
        case_name=args.case,
        output_dir=output_dir,
        preview=args.preview,
        fps=args.fps,
        overrides=overrides,
        animation=not args.no_animation,
    )
    return 0 if args.preview or run["diagnostics"]["converged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
