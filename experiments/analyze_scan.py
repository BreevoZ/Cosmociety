import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("outputs/convection_scan.csv")
DEFAULT_OUTPUT = Path("outputs/convection_scan_report.txt")


FLOAT_FIELDS = {
    "opacity_temperature_power",
    "opacity_density_power",
    "opacity_power",
    "radiative_density_power",
    "convective_max_diffusivity",
    "convective_nabla_ad",
    "final_delta",
    "temperature_contrast",
    "temperature_center",
    "temperature_surface",
    "convective_fraction",
    "convective_inner_radius",
    "convective_outer_radius",
    "max_radiative_diffusivity",
    "max_convective_diffusivity",
    "max_total_diffusivity",
    "max_luminosity",
    "max_nabla_rad",
}

INT_FIELDS = {
    "run",
    "converged_step",
    "num_regions",
    "convective_region_count",
}

BOOL_FIELDS = {
    "converged",
    "has_convective_core",
    "has_convective_envelope",
}


def parse_value(key, value):
    if value == "" or value is None:
        return None
    if key in BOOL_FIELDS:
        return value == "True"
    if key in INT_FIELDS:
        try:
            return int(value)
        except ValueError:
            return None
    if key in FLOAT_FIELDS:
        try:
            return float(value)
        except ValueError:
            return None
    return value


def read_rows(path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = [
            {key: parse_value(key, value) for key, value in row.items()}
            for row in reader
        ]
    for row in rows:
        row.setdefault("opacity_temperature_power", row.get("opacity_power"))
        row.setdefault("opacity_density_power", row.get("radiative_density_power"))
    return rows


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 1000 or (value != 0 and abs(value) < 1e-2):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def runs(rows):
    return ", ".join(str(row["run"]) for row in rows) if rows else "none"


def group_by(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(key)].append(row)
    return dict(sorted(groups.items()))


def trend_table(rows, key):
    lines = []
    for value, group in group_by(rows, key).items():
        lines.append(
            "  "
            f"{key}={fmt(value)}: "
            f"mean conv={fmt(mean(row['convective_fraction'] for row in group))}, "
            f"mean inner={fmt(mean(row['convective_inner_radius'] for row in group))}, "
            f"mean Tc/Ts={fmt(mean(row['temperature_contrast'] for row in group))}, "
            f"open={sum(not row['converged'] for row in group)}/{len(group)}"
        )
    return lines


def top_rows(rows, key, reverse=True, limit=5):
    usable = [row for row in rows if row.get(key) is not None]
    return sorted(usable, key=lambda row: row[key], reverse=reverse)[:limit]


def build_report(rows):
    converged = [row for row in rows if row["converged"]]
    open_rows = [row for row in rows if not row["converged"]]
    core_rows = [row for row in rows if row["has_convective_core"]]
    envelope_rows = [row for row in rows if row["has_convective_envelope"]]
    multi_region_rows = [row for row in rows if (row["num_regions"] or 0) > 1]

    lines = [
        "Cosmociety Convection Scan Report",
        "=" * 36,
        "",
        f"runs: {len(rows)}",
        f"converged: {len(converged)}",
        f"open: {len(open_rows)}",
        f"convective core runs: {runs(core_rows)}",
        f"convective envelope runs: {runs(envelope_rows)}",
        f"multi-region runs: {runs(multi_region_rows)}",
        "",
        "Multi-Region Details",
        "-" * 20,
    ]

    if multi_region_rows:
        for row in multi_region_rows:
            lines.append(
                f"  run {row['run']}: regions={row['convective_regions']} "
                f"p={fmt(row.get('opacity_temperature_power') or row.get('opacity_power'))} "
                f"Dconv_max={fmt(row['convective_max_diffusivity'])} "
                f"nabla_ad={fmt(row['convective_nabla_ad'])}"
            )
    else:
        lines.append("  none")

    lines.extend(
        [
            "",
            "Open Runs",
            "-" * 9,
        ]
    )
    if open_rows:
        for row in open_rows:
            lines.append(
                f"  run {row['run']}: delta={fmt(row['final_delta'])} "
                f"p={fmt(row.get('opacity_temperature_power') or row.get('opacity_power'))} "
                f"Dconv_max={fmt(row['convective_max_diffusivity'])} "
                f"nabla_ad={fmt(row['convective_nabla_ad'])}"
            )
    else:
        lines.append("  none")

    lines.extend(
        [
            "",
            "Parameter Trends",
            "-" * 16,
            "opacity_temperature_power:",
            *trend_table(rows, "opacity_temperature_power"),
            "",
            "convective_max_diffusivity:",
            *trend_table(rows, "convective_max_diffusivity"),
            "",
            "convective_nabla_ad:",
            *trend_table(rows, "convective_nabla_ad"),
            "",
            "Largest Convective Fractions",
            "-" * 28,
        ]
    )

    for row in top_rows(rows, "convective_fraction"):
        lines.append(
            f"  run {row['run']}: conv={fmt(row['convective_fraction'])} "
            f"regions={row['convective_regions']} "
            f"Tc/Ts={fmt(row['temperature_contrast'])}"
        )

    lines.extend(
        [
            "",
            "Largest Temperature Contrasts",
            "-" * 29,
        ]
    )
    for row in top_rows(rows, "temperature_contrast"):
        lines.append(
            f"  run {row['run']}: Tc/Ts={fmt(row['temperature_contrast'])} "
            f"conv={fmt(row['convective_fraction'])} "
            f"regions={row['convective_regions']}"
        )

    lines.extend(
        [
            "",
            "Interpretation Hints",
            "-" * 20,
            "- multi-region runs indicate separated convective zones, usually core + envelope.",
            "- open runs are useful for trends, but should not be treated as strict equilibrium.",
            "- increasing opacity_temperature_power usually moves the convective envelope inward.",
            "- increasing Dconv_max tends to lower Tc/Ts by strengthening heat transport.",
        ]
    )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Analyze a convection scan CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_rows(args.input)
    report = build_report(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report)
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
