# Cosmociety

Cosmociety is a toy computational physics framework for studying how simple
local transport rules produce emergent large-scale structure.

The first playground is stellar structure, not because the goal is only to
model stars, but because stars are clean self-organizing transport systems:
localized heating, outward energy flow, opacity, cooling boundaries, and
instabilities can generate distinct transport regimes without hard-coding
where those regimes should be.

This project intentionally favors interpretable toy physics over premature
realism. The code is meant to grow like a small research codebase: add one
physical effect at a time, isolate it, scan it, diagnose it, and only then make
the next rule more sophisticated.

## Current Model

The current model evolves a one-dimensional radial temperature profile on a
normalized grid from the center to the surface:

```text
r = 0                              r = 1
center                             surface
hot core source  ->  transport  -> radiative cooling boundary
```

At each timestep, the model combines:

- a core-localized heating source,
- a density-dependent heat capacity,
- radiative diffusion from a toy opacity law,
- optional convection triggered by local instability criteria,
- a surface cooling flux proportional to `T_surface^4 - T_space^4`.

The important design choice is that convective regions are not prescribed.
They emerge from transport rules and diagnostics.

## Quick Start

Create an environment and install the runtime dependencies.

On macOS/Linux, including Linux inside Windows through WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy matplotlib pillow
```

If you are using native Windows PowerShell instead of WSL:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy matplotlib pillow
```

Run a short preview case:

```bash
python3 main.py --preview
```

In native Windows PowerShell, use `python` or `py -3` in place of `python3`:

```powershell
python main.py --preview
```

Run the default case to equilibrium:

```bash
python3 main.py
```

Generated figures, animations, summaries, and CSV scans are written under
`outputs/`. The repository keeps only `outputs/.gitkeep`; run products are not
tracked.

## Command Line

Run one named demo case:

```bash
python3 main.py --case baseline_envelope
```

Available demo cases:

- `baseline_envelope`: radiative interior with a convective outer envelope.
- `dual_convection`: small convective core plus convective outer envelope.
- `strong_envelope`: deeper convective envelope with stronger mixing.
- `no_convection`: radiative-only reference run.

Useful overrides:

```bash
python3 main.py \
  --case baseline_envelope \
  --convective-transport excess \
  --surface-cooling 10 \
  --convective-max-diffusivity 1e-3 \
  --opacity-temperature-power 10 \
  --opacity-density-power 0.1
```

The convective transport law can be:

- `excess`: transport only the superadiabatic temperature drop.
- `diffusive`: treat convection as an additional diffusive channel.

Run all demo cases in preview mode:

```bash
python3 experiments/run_demo_cases.py
```

Run all demo cases to equilibrium:

```bash
python3 experiments/run_demo_cases.py --equilibrium
```

Scan convection parameters:

```bash
python3 experiments/scan_convection.py --quick
python3 experiments/scan_convection.py
```

Compare transport laws:

```bash
python3 experiments/scan_transport_law.py
```

Analyze a convection scan:

```bash
python3 experiments/analyze_scan.py \
  --input outputs/convection_scan.csv \
  --output outputs/convection_scan_report.txt
```

## Project Layout

```text
cosmociety/
  grid.py          normalized radial grid
  profiles.py      source, density, and initial temperature profiles
  opacity.py       toy opacity and radiative diffusivity laws
  convection.py    convection criteria and convective flux laws
  transport.py     flux-form thermal diffusion step
  equilibrium.py   relaxation loop that assembles the physics
  diagnostics.py   scalar summaries, regime labels, and convective regions
  visualize.py     static plots
  animation.py     relaxation animations

experiments/
  demo_cases.py          named parameter sets
  run_demo_cases.py      batch runner for demo cases
  scan_convection.py     opacity/convection parameter scan
  scan_transport_law.py  comparison of convective transport laws
  analyze_scan.py        text report from scan CSV output

main.py           command-line entry point for one case
outputs/          local generated results, ignored by git
```

## Development Philosophy

- Build incrementally, like a real research codebase.
- Avoid hard-coded stellar zones.
- Let structure emerge from local transport physics.
- Prefer interpretable toy rules over hidden realism.
- Keep every new physical effect isolated enough to scan and diagnose.
- Treat plots, summaries, and regime classification as part of the model.

The guiding question is not "Can this reproduce a real star yet?" but:

> What kinds of large-scale transport regimes emerge from simple local rules?

## Notes

This is not currently a packaged Python project. Run scripts from the repository
root so local imports resolve correctly.

The numerical scheme is explicit and intentionally simple. Parameter choices
can affect stability and convergence, so scans should track both converged and
open runs.
