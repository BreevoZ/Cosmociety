import numpy as np
import pytest

from cosmociety.animation import animate_relaxation
from cosmociety.convection import (
    convective_diffusivity_from_schwarzschild,
)
from cosmociety.equilibrium import relax_to_equilibrium
from cosmociety.opacity import radiative_diffusivity_from_state

# A small, fast case (n=25) that reaches equilibrium in well under a second,
# used both as a convergence smoke test and as a regression baseline: the
# simulation is fully deterministic given these parameters, so any change to
# the physics laws or the timestepping loop that isn't intentional should
# show up here as a changed final temperature profile.
GOLDEN_PARAMS = dict(
    n=25,
    max_steps=200_000,
    tolerance=1e-6,
    save_every=5000,
    opacity_temperature_power=10.0,
    convective_nabla_ad=0.4,
    convective_max_diffusivity=1e-3,
)

GOLDEN_TEMPERATURE = np.array(
    [
        1.08989942, 1.08989942, 1.07222504, 1.04377318, 1.009691,
        0.97581437, 0.94636233, 0.92124999, 0.89754207, 0.8725087,
        0.84461066, 0.81317929, 0.77800381, 0.73910382, 0.69662008,
        0.65074385, 0.60165903, 0.54950328, 0.49435353, 0.43623106,
        0.37511733, 0.31097236, 0.22208721, 0.14546952, 0.09051259,
    ]
)
GOLDEN_CONVERGED_STEP = 30545


def test_relax_to_equilibrium_matches_golden_baseline():
    result = relax_to_equilibrium(**GOLDEN_PARAMS)
    assert result["converged_step"] == GOLDEN_CONVERGED_STEP
    assert result["temperature"] == pytest.approx(GOLDEN_TEMPERATURE, rel=1e-6, abs=1e-9)


def test_relax_to_equilibrium_basic_invariants():
    result = relax_to_equilibrium(**GOLDEN_PARAMS)
    T = result["temperature"]
    assert result["converged_step"] is not None
    assert np.all(np.isfinite(T))
    assert T[0] == T[1]  # center symmetry
    assert T[0] > T[-1]  # core hotter than surface
    assert np.all(result["radiative_diffusivity"] > 0)
    assert np.all(result["convective_diffusivity"] >= 0)
    assert np.all(result["density"] > 0)


@pytest.mark.parametrize("convective_transport", ["excess", "diffusive"])
def test_relax_to_equilibrium_runs_for_both_transport_modes(convective_transport):
    result = relax_to_equilibrium(
        n=20,
        max_steps=20_000,
        tolerance=1e-5,
        save_every=2000,
        convective_transport=convective_transport,
    )
    assert np.all(np.isfinite(result["temperature"]))


def test_evaluate_convection_matches_direct_recomputation():
    # equilibrium.py's returned diffusivities should be exactly what calling
    # the same convection.py/opacity.py functions directly on the final
    # state produces -- this is the "single source of truth" invariant that
    # a duplicated/hand-mirrored implementation (like animation.py used to
    # have) could silently violate.
    result = relax_to_equilibrium(**GOLDEN_PARAMS)

    D_rad = radiative_diffusivity_from_state(
        temperature=result["temperature"],
        density=result["density"],
        diffusivity_scale=result["radiative_diffusivity_scale"],
        opacity_temperature_power=result["opacity_temperature_power"],
        opacity_density_power=result["opacity_density_power"],
        diffusivity_floor=result["radiative_diffusivity_floor"],
    )
    assert D_rad == pytest.approx(result["radiative_diffusivity"])

    D_conv, _, _ = convective_diffusivity_from_schwarzschild(
        source=result["source"],
        radius=result["r"],
        temperature=result["temperature"],
        pressure=result["pressure"],
        diffusivity=D_rad,
        dr=result["r"][1] - result["r"][0],
        nabla_ad=result["convective_nabla_ad"],
        strength=result["convective_strength"],
        max_diffusivity=result["convective_max_diffusivity"],
    )
    assert D_conv == pytest.approx(result["convective_diffusivity"])


def test_animate_relaxation_runs_on_a_short_history(tmp_path):
    # Smoke test for the exact code path touched by the animation.py
    # refactor that removed its duplicated opacity/convection formulas in
    # favor of calling cosmociety.opacity/convection/transport directly. A
    # broken wiring (wrong argument, shape mismatch, missing result key)
    # should fail here immediately rather than only showing up as a subtly
    # wrong GIF nobody looked closely at.
    result = relax_to_equilibrium(
        n=15,
        max_steps=3000,
        tolerance=1e-6,
        save_every=500,
    )
    save_path = tmp_path / "relaxation.gif"
    animate_relaxation(result, save_path=str(save_path), fps=5)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
