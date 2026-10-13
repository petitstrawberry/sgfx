#!/bin/sh
set -eu
recipe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$recipe_dir/native-config.py" install "$PWD" "$1"
