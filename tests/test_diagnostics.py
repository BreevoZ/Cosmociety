import numpy as np
import pytest

from cosmociety.diagnostics import (
    classify_regime,
    classify_structural_regime,
    convective_regions,
    format_regions,
    format_summary,
    summarize_result,
)


def test_convective_regions_empty_when_nothing_active():
    r = np.linspace(0.0, 1.0, 10)
    active = np.zeros(10, dtype=bool)
    assert convective_regions(r, active) == []


def test_convective_regions_finds_single_contiguous_span():
    r = np.linspace(0.0, 1.0, 10)
    active = np.zeros(10, dtype=bool)
    active[3:6] = True
    regions = convective_regions(r, active)
    assert regions == [(r[3], r[5])]


def test_convective_regions_finds_multiple_spans():
    r = np.linspace(0.0, 1.0, 10)
    active = np.zeros(10, dtype=bool)
    active[0:2] = True  # core
    active[7:10] = True  # envelope
    regions = convective_regions(r, active)
    assert regions == [(r[0], r[1]), (r[7], r[9])]


def test_format_regions():
    assert format_regions([]) == "none"
    assert format_regions([(0.0, 0.08), (0.54, 1.0)]) == "0.000-0.080;0.540-1.000"


@pytest.mark.parametrize(
    "regions,has_core,has_envelope,expected",
    [
        ([], False, False, "radiative_only"),
        ([(0.0, 1.0)], True, True, "global_convection"),
        ([(0.0, 0.1), (0.6, 1.0)], True, True, "dual_convection"),
        ([(0.0, 0.1)], True, False, "convective_core"),
        ([(0.6, 1.0)], False, True, "convective_envelope"),
        ([(0.3, 0.5)], False, False, "internal_convection"),
        ([(0.2, 0.3), (0.5, 0.6)], False, False, "multiple_internal_convection"),
    ],
)
def test_classify_structural_regime(regions, has_core, has_envelope, expected):
    assert classify_structural_regime(regions, has_core, has_envelope) == expected


def test_classify_regime_prefixes_open_when_not_converged():
    assert classify_regime("convective_core", converged=True) == "convective_core"
    assert classify_regime("convective_core", converged=False) == "open_convective_core"


def _fake_result(active_mask, converged_step=5):
    n = len(active_mask)
    r = np.linspace(0.0, 1.0, n)
    convective_diffusivity = np.where(active_mask, 1e-4, 0.0)
    return {
        "r": r,
        "temperature": np.linspace(1.0, 0.1, n),
        "convective_diffusivity": convective_diffusivity,
        "radiative_diffusivity": np.full(n, 0.02),
        "diffusivity": np.full(n, 0.02) + convective_diffusivity,
        "luminosity": np.linspace(0.0, 1.0, n - 1),
        "density": np.linspace(1.0, 0.1, n),
        "deltas": np.array([1e-3, 1e-6]),
        "timesteps": np.array([1e-4, 1e-4]),
        "converged_step": converged_step,
        "geometry": "spherical",
        "convective_criterion": "schwarzschild",
        "convective_transport": "excess",
        "surface_cooling": 10.0,
        "opacity_temperature_power": 10.0,
        "opacity_density_power": 0.1,
        "opacity_power": 10.0,
        "radiative_density_power": 0.1,
    }


def test_summarize_result_converged_envelope_case():
    n = 10
    active = np.zeros(n, dtype=bool)
    active[7:] = True  # envelope only
    summary = summarize_result(_fake_result(active))

    assert summary["converged"] is True
    assert summary["structural_regime"] == "convective_envelope"
    assert summary["regime"] == "convective_envelope"
    assert summary["has_convective_envelope"] is True
    assert summary["has_convective_core"] is False
    assert summary["temperature_contrast"] == pytest.approx(1.0 / 0.1)


def test_summarize_result_open_when_not_converged():
    n = 10
    active = np.zeros(n, dtype=bool)
    summary = summarize_result(_fake_result(active, converged_step=None))
    assert summary["converged"] is False
    assert summary["regime"] == "open_radiative_only"


def test_format_summary_runs_on_a_real_summary():
    n = 10
    active = np.zeros(n, dtype=bool)
    active[:2] = True
    text = format_summary(summarize_result(_fake_result(active)))
    assert "regime:" in text
    assert "converged: yes, step 5" in text
