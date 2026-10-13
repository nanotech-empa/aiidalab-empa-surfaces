#!/usr/bin/env python3
"""Fail if any notebook cell stores `source` as a single string.

nbformat allows a cell's `source` to be either a string or a list of
one-line strings, but only the list form diffs line by line. Jupyter
always writes the list form; this check catches tools that write the
flattened form instead, since neither ruff nor check-json does.
"""

import json
import sys


def find_flattened_cells(path):
    with open(path, encoding="utf-8") as fh:
        notebook = json.load(fh)
    return [
        index
        for index, cell in enumerate(notebook.get("cells", []))
        if isinstance(cell.get("source"), str)
    ]


def main(argv):
    failed = False
    for path in argv:
        flattened = find_flattened_cells(path)
        if flattened:
            failed = True
            print(
                f"{path}: cell(s) {flattened} have a string-valued 'source', "
                "expected a list of lines."
            )
    if failed:
        print(
            "\nReflow with: "
            'python -c "import nbformat,sys; '
            "nb = nbformat.read(sys.argv[1], as_version=nbformat.NO_CONVERT); "
            'nbformat.write(nb, sys.argv[1])" <path>'
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
