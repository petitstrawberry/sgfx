#!/usr/bin/env python3
"""Write a development ICD manifest pointing to an existing built library."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("library", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
library = args.library.resolve(strict=True)
if not library.is_file():
    parser.error("library must be a file")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({
    "file_format_version": "1.0.0",
    "ICD": {"library_path": str(library), "api_version": "1.0.0"},
}, indent=2) + "\n")
print(args.output.resolve())
