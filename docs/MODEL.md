# Model and numerical methods

This document describes the implementation in this repository. It is a record
of the toy model, not an assertion that these closures reproduce stellar physics.
Radius, temperature, density, time, and transport coefficients are normalized;
there is no physical-unit calibration.

## Governing thermal equation

For spherical geometry and outward-positive flux,

\[
C(r)\frac{\partial T}{\partial t}
= -\frac{1}{r^2}\frac{\partial}{\partial r}(r^2 F) + S(r),
\qquad F = F_{\mathrm{rad}} + F_{\mathrm{conv}}.
\]

The Cartesian alternative omits the radial area factors. `transport.py` performs
an explicit step using interface fluxes and cell-centered temperatures. The
normalized luminosity diagnostic is `r_interface**2 * flux`; it omits `4*pi`.

## Prescribed profiles and constitutive rules

| Quantity | Rule | Implementation |
| --- | --- | --- |
| Grid | Uniform `r` from 0 to 1, at least 3 points | `grid.radial_grid` |
| Heating | `S = strength * exp[-(r/width)^2]`; solver uses strength 1, width 0.12 | `profiles.core_source` |
| Initial temperature | `T = 0.1 + 0.9*(1-r^2)` | `profiles.initial_temperature` |
| Density | `rho_s + (rho_c-rho_s)*(1-r^2)^concentration` | `profiles.density_profile` |
| Heat capacity | `C = rho` | `equilibrium.relax_to_equilibrium` |
| Pressure proxy | `P = rho^m T`; default `m=3` | `equilibrium.relax_to_equilibrium` |
| Opacity | `kappa = (rho/rho_s)^q T^(-p)` | `opacity.opacity_from_state` |
| Radiative coefficient | `D_rad = max(D0/kappa, D_floor)` | `opacity.radiative_diffusivity_from_state` |

Density does not evolve. Pressure is recomputed algebraically from the current
temperature and the prescribed density, without solving hydrostatic balance.

## Radiation and convection

Radiative interface flux uses the arithmetic mean of neighboring coefficients:

\[
F_{\rm rad,\,i+1/2} =
-\frac{D_i+D_{i+1}}{2}\frac{T_{i+1}-T_i}{\Delta r}.
\]

Three independent activation criteria are available through the Python API:

- `gradient`: the outward temperature drop exceeds a fixed threshold.
- `radiative`: the enclosed source implies a radiation-only required gradient
  greater than a fixed threshold.
- `schwarzschild` (default): the source-implied gradient is expressed as a
  logarithmic temperature/pressure gradient and exceeds `nabla_ad`.

For the default criterion, the code approximates

\[
L_{\rm req,\,i+1/2} = \sum_{j\le i} S_j r_j^2 \Delta r,
\quad G_{\rm req} = \frac{L_{\rm req}}{r_{i+1/2}^2 D_{\rm rad,\,i+1/2}},
\quad \nabla_{\rm rad} = \frac{P_{i+1/2}G_{\rm req}}{T_{i+1/2}(-\Delta P/\Delta r)}.
\]

Small denominator floors are used in the implementation. The positive excess
above the threshold is scaled by `convective_strength`, capped by
`convective_max_diffusivity`, and averaged from interfaces to grid points.
This is a source-demand proxy even during transient runs; it is not a solution
of a dynamical fluid instability.

The two transport laws are separate from activation:

- `diffusive`: `F_conv = D_conv,interface * (-Delta T/Delta r)`.
- `excess` (default):

\[
G_{\rm ad} = \nabla_{\rm ad}\frac{T_{i+1/2}}{P_{i+1/2}}
\max(-\Delta P/\Delta r,0),
\qquad F_{\rm conv} = D_{\rm conv,\,i+1/2}
\max(-\Delta T/\Delta r - G_{\rm ad},0).
\]

The excess law only transports the superadiabatic drop. Its magnitude is
bounded by diffusion of the same outward drop. An activated region with
`D_conv > 0` can still have zero excess flux. Inspect the returned
`convective_flux` array separately from the activation mask.

## Boundary and timestep treatment

At each step, `transport.diffusion_step` computes flux divergence and heating,
sets center symmetry with `T[0] = T[1]`, then applies surface cooling:

\[
F_{\rm surface} = a(T_{\rm surface}^4 - T_{\rm space}^4).
\]

Cooling is evaluated on the updated surface temperature and removed from the
outer cell as `dt*F_surface/(C_surface*dr)`. A floor enforces `T >= T_space`.
This sequential treatment and the floor matter when assessing conservation.

The relaxation loop limits the user timestep using

\[
\Delta t_{\rm step} = \min\left(\Delta t_{\rm user},
\frac{s\Delta r^2}{\max[(D_{\rm rad}+D_{\rm conv})/C]}\right).
\]

The combined coefficient bounds both transport modes. This is the implemented
diffusion-based limit; it does not by itself establish nonlinear stability
for every cooling strength, geometry, or parameter combination. Non-finite
temperatures raise an error.

## Stopping and diagnostics

The solver stops when `max(abs(T_new - T_old)) < tolerance` or reaches
`max_steps`. `converged_step` is the **zero-based** loop index; `steps_completed`
in exported metadata is the number of steps actually taken.

Preview mode sets `tolerance=-1`, intentionally disabling early stopping.
A run can request equilibrium but fail to meet the tolerance; its final
artifacts remain available, and command-line tools return exit code 2.

`convective_regions` groups adjacent grid points with positive convective
coefficient. The core/envelope flags use proximity to the second/penultimate
grid point. These are resolution-dependent classification conventions. The
active fraction is a count fraction of grid points, not a spherical-volume
fraction. `open_` distinguishes un-converged snapshots from runs that passed
the stopping test.

`history` contains the initial temperature and periodic saved snapshots; an
un-converged run can finish between snapshots. Animation adds the actual final
temperature for display without changing solver history or numerical outputs.
Animation activation bands use interface masks, while scalar region diagnostics
use grid-point masks, so their edges can differ by roughly one grid interval.

## Reproducibility and validation boundaries

JSON run records contain the full solver signature with defaults applied,
scalar diagnostics, Python/package versions, completed steps, and summed
timesteps. The NPZ exports every NumPy array from the result without object
pickles. Parameters do not encode hard-coded source rules; preserve the matching
source version as well. No random numbers enter the simulation.

The golden regression is a small deterministic case, not an observational
benchmark. Transport tests establish specific discrete invariants under their
tested conditions, not an end-to-end energy budget for the complete model.
Further evidence needed before stronger equilibrium or regime claims:

1. A source-to-boundary luminosity balance and thermal-energy residual that
   accounts for the implemented boundary and temperature-floor treatment.
2. Grid and timestep refinement, including robust region-boundary comparisons.
3. Sensitivity to pressure/density proxies, diffusivity floor, and cooling.
4. Controlled comparisons that vary one parameter or transport law at a time.

The code currently has no self-consistent gravitational field, hydrostatic
structure solve, dynamical density, fluid velocity, nuclear network, realistic
opacity table, or observational validation.
