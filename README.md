# Cosmociety

A small Python project for experimenting with radiative diffusion and convection
in a one-dimensional, stellar-inspired thermal model. Convective regions follow
local transport rules rather than being assigned in advance.

## Getting started

Requires Python 3.10 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python main.py --preview
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
The runtime dependencies are NumPy, Matplotlib, and Pillow; the `dev` extra adds
pytest and the package build tool.

Runs write figures, a GIF, a text summary, and JSON/NPZ data under
`outputs/cases/<case>/`. Use `--no-animation` to skip GIF rendering.
The installed `cosmociety` command and `python -m cosmociety` also work outside
the repository.

## Model

The solver evolves a radial temperature profile with a core-localized heat
source, prescribed density, radiative diffusion, optional convection, and a
radiative cooling boundary. The toy opacity law is
`kappa = (rho/rho_surface)^q * T^(-p)`. Convection can transport either the full
temperature gradient (`diffusive`) or only its superadiabatic excess (`excess`).

All quantities are normalized. Density is fixed and pressure is an algebraic
proxy; the model does not solve hydrostatic structure or calibrated stellar
physics. See [the model notes](docs/MODEL.md) for equations, numerical methods,
and diagnostic definitions.

## Experiments

| Case | Purpose |
| --- | --- |
| `baseline_envelope` | Outer convective envelope |
| `dual_convection` | Separated inner and outer active regions |
| `strong_envelope` | Stronger temperature dependence of opacity |
| `no_convection` | Radiative-only reference |

Case names describe parameter presets. Check the measured `regime` in the
output: `open_` means the stopping tolerance was not reached. Preview mode
intentionally disables early stopping. The stopping test measures the change
per timestep; independent energy-balance and resolution studies remain needed.

```bash
# Request convergence for a case
python main.py --case baseline_envelope

# Compare with ordinary diffusive convection
python main.py --preview --convective-transport diffusive \
  --output-dir outputs/diffusive_preview

# Run all presets in preview mode
python experiments/run_demo_cases.py

# Run a small parameter scan and analyze its output
python experiments/scan_convection.py --quick
python experiments/analyze_scan.py --input outputs/convection_scan_quick.csv \
  --output outputs/convection_scan_quick_report.txt
```

`python main.py --help` lists parameter overrides, including `--n` and
`--max-steps`. A run that exhausts its budget before convergence still saves its
final state and exits with status 2. Previews exit with status 0.

The JSON records all solver arguments, scalar diagnostics, and package versions.
The NPZ contains numerical arrays and can be opened with
`numpy.load(path, allow_pickle=False)`. Use the matching source version to
reproduce a run from its saved parameters.

## Development

```bash
python -m pytest
python -m build
```

Tests cover analytic rules, transport and boundary invariants, convection
thresholds, regime classification, a fixed temperature-profile regression,
and saved-result replay. CI checks tests and package installation.

```text
cosmociety/    model, solver, diagnostics, plotting, and CLI
experiments/  demo runners and parameter scans
tests/        deterministic tests
docs/         model and numerical-method notes
outputs/      generated results, ignored by Git
```

Generated results stay under `outputs/` and out of version control.
