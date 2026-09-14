import json

import numpy as np

from cosmociety.artifacts import export_result, resolved_parameters
from cosmociety.diagnostics import summarize_result
from cosmociety.equilibrium import relax_to_equilibrium


def test_export_can_reproduce_result_without_implicit_defaults(tmp_path):
    params = resolved_parameters({"n": 12, "max_steps": 8, "save_every": 3, "enable_convection": False})
    result = relax_to_equilibrium(**params)
    export_result(result, params, summarize_result(result), tmp_path, "_test", "no_convection", "equilibrium", 0.0)
    metadata = json.loads((tmp_path / "run_test.json").read_text())
    assert metadata["summary"]["regime"] == "open_radiative_only"
    assert metadata["summary"]["convective_inner_radius"] is None
    assert metadata["steps_completed"] == 8
    assert metadata["parameters"]["dt"] == 1e-5
    replay = relax_to_equilibrium(**metadata["parameters"])
    with np.load(tmp_path / "profiles_test.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["temperature"], replay["temperature"])
        np.testing.assert_array_equal(arrays["flux"], replay["flux"])
        assert all(arrays[key].dtype != object for key in arrays.files)
