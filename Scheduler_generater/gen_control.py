#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from controlgen.emit_sv import write_outputs
from controlgen.model import parse_model
from controlgen.validate import validate_spec


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = SCRIPT_DIR / "template" / "conv_acc_control.template.json"


def default_template_spec() -> dict:
    return {
        "InstructionWidth": 128,
        "OpcodeWidth": 8,
        "RegAddrWidth": 8,
        "PayloadWidth": 120,
        "System": {
            "Nop": {},
            "Sync": {"Deps": ["read_u", "conv_u", "write_back_u"]},
        },
        "Pingpongs": {
            "ipp": {},
            "opp": {},
        },
        "Modules": {
            "read_u": {
                "CommandOpcodeWidth": 1,
                "Command": {
                    "to_ipp": {
                        "Args": [
                            {"Name": "base_addr", "Width": 32, "Default": 0},
                            {"Name": "length", "Width": 32},
                        ],
                        "Using": ["ipp"],
                    },
                    "to_opp": {
                        "Args": [
                            {"Name": "base_addr", "Width": 32, "Default": 0},
                            {"Name": "length", "Width": 32},
                        ],
                        "Using": ["opp"],
                    },
                },
                "Regs": {},
            }
        },
    }


def load_template_spec() -> dict:
    if TEMPLATE_PATH.exists():
        return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return default_template_spec()


def load_spec(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate scheduler control RTL from a JSON control-plane spec.")
    parser.add_argument("spec", nargs="?", help="input JSON specification")
    parser.add_argument("outdir", nargs="?", help="output directory")
    parser.add_argument("--check-only", action="store_true", help="validate the spec without writing files")
    parser.add_argument("--emit-template", metavar="FILE", help="write a small template JSON spec")
    args = parser.parse_args()

    if args.emit_template:
        Path(args.emit_template).write_text(json.dumps(load_template_spec(), indent=2) + "\n", encoding="utf-8")
        return 0

    if not args.spec:
        parser.error("spec is required unless --emit-template is used")
    spec_path = Path(args.spec)
    spec = load_spec(spec_path)

    try:
        validate_spec(spec)
        if args.check_only:
            print("spec OK")
            return 0
        if not args.outdir:
            parser.error("outdir is required unless --check-only is used")
        model = parse_model(spec)
        repo_root = Path(__file__).resolve().parents[1]
        written = write_outputs(model, Path(args.outdir), repo_root)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for name in written:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
