import json

from cosmociety.cli import main


def test_open_equilibrium_run_preserves_diagnostics_and_signals_failure(tmp_path):
    exit_code = main(["--case", "no_convection", "--n", "12", "--max-steps", "4", "--no-animation", "--output-dir", str(tmp_path)])
    assert exit_code == 2
    metadata = json.loads((tmp_path / "run_excess.json").read_text())
    assert metadata["summary"]["converged"] is False
    assert metadata["summary"]["regime"] == "open_radiative_only"
    assert (tmp_path / "profiles_excess.npz").exists()
    assert "converged: no" in (tmp_path / "summary_excess.txt").read_text()


def test_preview_accepts_explicit_budget_and_succeeds_without_convergence(tmp_path):
    exit_code = main(["--preview", "--n", "12", "--max-steps", "4", "--no-animation", "--output-dir", str(tmp_path)])
    assert exit_code == 0
    metadata = json.loads((tmp_path / "run_excess_preview.json").read_text())
    assert metadata["parameters"]["n"] == 12
    assert metadata["steps_completed"] == 4
    assert metadata["parameters"]["tolerance"] < 0
