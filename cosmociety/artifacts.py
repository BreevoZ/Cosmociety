"""Portable numerical results and the information needed to reproduce them."""

import inspect
import json
import math
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from .equilibrium import relax_to_equilibrium


def resolved_parameters(overrides: dict) -> dict:
    """Include solver defaults so a future default change cannot alter a replay."""
    bound = inspect.signature(relax_to_equilibrium).bind(**overrides)
    bound.apply_defaults()
    return dict(bound.arguments)


def json_safe(value):
    """Encode missing numerical diagnostics as JSON null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def runtime_provenance() -> dict:
    packages = {}
    for name in ("cosmociety", "numpy", "matplotlib", "pillow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "source checkout (not installed)"

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def export_result(
    result: dict,
    parameters: dict,
    summary: dict,
    output_dir: Path,
    suffix: str,
    case_name: str,
    mode: str,
    elapsed_seconds: float,
) -> None:
    """Save scalar metadata and all numerical arrays without pickled objects."""
    metadata_path = output_dir / f"run{suffix}.json"
    data_path = output_dir / f"profiles{suffix}.npz"
    write_json(metadata_path, {
        "schema_version": 1,
        "case": case_name,
        "mode": mode,
        "parameters": parameters,
        "summary": summary,
        "elapsed_seconds": elapsed_seconds,
        "steps_completed": len(result["deltas"]),
        "simulated_time": float(np.sum(result["timesteps"])),
        "provenance": runtime_provenance(),
    })
    np.savez_compressed(data_path, **{
        key: value for key, value in result.items() if isinstance(value, np.ndarray)
    })
