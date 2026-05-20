import argparse
from pathlib import Path

from cosmociety.equilibrium import relax_to_equilibrium
from cosmociety.diagnostics import format_summary, summarize_result
from cosmociety.visualize import plot_equilibrium, plot_transport_diagnostics
from cosmociety.animation import animate_relaxation
from experiments.demo_cases import case_names, get_case


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
    return parser.parse_args()


def main():
    args = parse_args()
    case = get_case(args.case)
    output_dir = args.output_dir or Path("outputs") / "cases" / args.case
    output_dir.mkdir(parents=True, exist_ok=True)

    params = dict(case["params"])
    suffix = ""
    mode = "equilibrium"
    if args.preview:
        mode = "preview"
        suffix = "_preview"
        params.update(
            {
                "n": 100,
                "max_steps": 60_000,
                "save_every": 3_000,
                "tolerance": -1.0,
            }
        )

    result = relax_to_equilibrium(**params)

    if result["converged_step"] is None and not args.preview:
        raise RuntimeError("Simulation did not reach equilibrium; skipping diagnostics.")

    plot_equilibrium(result, save_path=str(output_dir / f"radiative_equilibrium{suffix}.png"))
    plot_transport_diagnostics(
        result,
        save_path=str(output_dir / f"transport_diagnostics{suffix}.png"),
    )

    fps = args.fps if args.fps is not None else (10 if args.preview else 30)
    animate_relaxation(
        result,
        save_path=str(output_dir / f"radiative_relaxation{suffix}.gif"),
        fps=fps,
    )

    summary = format_summary(summarize_result(result))
    (output_dir / f"summary{suffix}.txt").write_text(
        f"case: {args.case}\n"
        f"mode: {mode}\n"
        f"description: {case['description']}\n\n"
        f"{summary}\n"
    )

    print(f"Case: {args.case}")
    print(f"Mode: {mode}")
    print(case["description"])
    print("Simulation complete.")
    print(f"Output directory: {output_dir}")
    print(f"Converged step: {result['converged_step']}")
    print(summary)


if __name__ == "__main__":
    main()
    
