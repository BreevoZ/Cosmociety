# Repository Guidelines

## Project Structure & Module Organization

Cosmociety is a small Python codebase for toy 1D stellar transport experiments.

- `main.py` is the command-line entry point for running one demo case.
- `cosmociety/` contains the core model: grids, profiles, opacity, convection, transport, equilibrium, diagnostics, plots, and animation.
- `experiments/` contains batch runners, parameter scans, and scan analysis scripts.
- `outputs/` is for generated figures, GIFs, summaries, and CSV files. Keep generated run products out of commits.
- `tests/` contains the pytest suite (one file per `cosmociety/` module, plus `test_equilibrium.py` for cross-module/regression checks).

Run scripts from the repository root so local imports resolve correctly.

## Build, Test, and Development Commands

Create a local environment and install runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy matplotlib pillow pytest
```

Run a short preview:

```bash
python3 main.py --preview
```

Run one named case to equilibrium:

```bash
python3 main.py --case baseline_envelope
```

Run all demo cases in preview mode:

```bash
python3 experiments/run_demo_cases.py
```

Run a quick convection scan:

```bash
python3 experiments/scan_convection.py --quick
```

Analyze a completed scan:

```bash
python3 experiments/analyze_scan.py --input outputs/convection_scan.csv
```

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation and clear snake_case names for modules, functions, variables, and parameters. Prefer small, focused functions that isolate one physical rule or diagnostic. Keep numerical formulas explicit and readable; add comments only where they clarify non-obvious physics or numerical choices.

This repository does not currently configure a formatter or linter. If adding one, document the exact command and avoid unrelated formatting churn.

## Testing Guidelines

Run the pytest suite (deterministic, ~5s):

```bash
python3 -m pytest
```

`test_equilibrium.py::test_relax_to_equilibrium_matches_golden_baseline` pins the exact converged temperature profile of a small, fast case; if you intentionally change a physics law, opacity/convection formula, or the timestepping loop, regenerate `GOLDEN_TEMPERATURE`/`GOLDEN_CONVERGED_STEP` there from the new output rather than loosening the tolerance. For changes to model behavior, also run:

```bash
python3 main.py --preview
python3 experiments/scan_convection.py --quick
```

New tests go under `tests/`, named `test_*.py`. Favor deterministic checks: analytic values for grid/profiles/opacity, conservation and boundary-condition invariants for `transport.diffusion_step`, threshold behavior and the "excess flux never exceeds ordinary diffusive flux" invariant for `convection.py`, and region/regime classification for `diagnostics.py`.

## Commit & Pull Request Guidelines

Recent commit messages are short and descriptive, for example `Organized output` and `Made radiative diffusivity dependent on density`. Use concise, action-oriented messages that explain the model or workflow change.

Pull requests should include a brief summary, commands run, output artifacts affected, and any expected changes in regimes, convergence, or scan results. Include plots or GIF references when visual diagnostics change.

## Configuration Tips

Matplotlib may try to write its cache outside the repository. If needed, set `MPLCONFIGDIR` to a writable local or temporary directory before running plotting commands.
