DEMO_CASES = {
    "baseline_envelope": {
        "description": "Radiative interior with a convective outer envelope.",
        "params": {
            "opacity_temperature_power": 10.0,
            "convective_nabla_ad": 0.4,
            "convective_max_diffusivity": 1e-3,
        },
    },
    "dual_convection": {
        "description": "Small convective core plus convective outer envelope.",
        "params": {
            "opacity_temperature_power": 8.0,
            "convective_nabla_ad": 0.35,
            "convective_max_diffusivity": 5e-4,
        },
    },
    "strong_envelope": {
        "description": "Deeper convective envelope with stronger convective mixing.",
        "params": {
            "opacity_temperature_power": 12.0,
            "convective_nabla_ad": 0.4,
            "convective_max_diffusivity": 1e-3,
        },
    },
    "no_convection": {
        "description": "Radiative-only reference run.",
        "params": {
            "enable_convection": False,
        },
    },
}


def case_names() -> list[str]:
    return sorted(DEMO_CASES)


def get_case(name: str) -> dict:
    try:
        return DEMO_CASES[name]
    except KeyError as exc:
        available = ", ".join(case_names())
        raise ValueError(f"unknown case {name!r}; available cases: {available}") from exc
