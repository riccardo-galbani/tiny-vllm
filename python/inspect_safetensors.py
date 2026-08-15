#!/usr/bin/env python3
"""
Inspect a .safetensors file header: prints name, dtype and shape for
every tensor. Uses only the Python standard library (json, struct) —
no external dependencies required.

Usage:
    # all tensors
    python3 inspect_safetensors.py model.safetensors

    # only Q projections (useful for checking the 2048x2048 shape mentioned in the README)
    python3 inspect_safetensors.py model.safetensors --filter q_proj

    # all tensors in layer 0 (to see the full structure of a single transformer block)
    python3 inspect_safetensors.py model.safetensors --layer 0

    # with offsets, to see where each tensor starts/ends in the data section
    python3 inspect_safetensors.py model.safetensors --offsets

    # only the __metadata__ contents, if present
    python3 inspect_safetensors.py model.safetensors --metadata
"""

import argparse
import json
import struct
import sys


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError(f"File too short to be a valid safetensors file: {path}")

        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json = f.read(header_size)
        return json.loads(header_json)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a .safetensors file header (name, dtype, shape, offsets for each tensor)."
    )
    parser.add_argument("model", help="Path to the .safetensors file to inspect")
    parser.add_argument(
        "--filter", "-f", default=None,
        help="Only show tensors whose name contains this substring (e.g. 'q_proj')"
    )
    parser.add_argument(
        "--layer", "-l", type=int, default=None,
        help="Only show tensors from layer N (e.g. --layer 0 for model.layers.0.*)"
    )
    parser.add_argument(
        "--offsets", action="store_true",
        help="Also show offsets (start/end byte) of each tensor in the data section"
    )
    parser.add_argument(
        "--metadata", action="store_true",
        help="Only show the __metadata__ contents, if present"
    )
    args = parser.parse_args()

    try:
        header = read_header(args.model)
    except FileNotFoundError:
        print(f"Error: file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing header: {e}", file=sys.stderr)
        sys.exit(1)

    metadata = header.pop("__metadata__", None)

    if args.metadata:
        print(json.dumps(metadata, indent=2, ensure_ascii=False) if metadata else "(no __metadata__ present)")
        return

    layer_prefix = f"model.layers.{args.layer}." if args.layer is not None else None

    count = 0
    for name, info in sorted(header.items()):
        if args.filter and args.filter not in name:
            continue
        if layer_prefix and not name.startswith(layer_prefix):
            continue

        dtype = info.get("dtype", "?")
        shape = info.get("shape", [])
        line = f"{name:60s} {dtype:8s} {str(shape)}"

        if args.offsets:
            offsets = info.get("data_offsets", [])
            line += f"  offsets={offsets}"

        print(line)
        count += 1

    print(f"\n--- {count} tensors shown (out of {len(header)} total in file) ---", file=sys.stderr)


if __name__ == "__main__":
    main()
