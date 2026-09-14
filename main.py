"""Backward-compatible source-checkout entry point."""

from cosmociety.cli import PREVIEW_PARAMS, main, parse_args, run_case


if __name__ == "__main__":
    raise SystemExit(main())
