# Command reference

Install the package with `python -m pip install -e .` and build the C library as
shown in the [README](../README.md). `maze --help` lists the top-level groups;
each group and command also supports `--help`.

| Command | Purpose |
| --- | --- |
| `maze --help` | List the available command groups. |
| `maze run md INPUT.yml` | Run molecular dynamics from a YAML input file. JSON files are also accepted. Add `-v` for debug logging. |

`maze plot --help` and `maze utils --help` expose research utilities. They are
not yet part of the first-use workflow: several plotters require historical
filename patterns or hard-coded data paths, `generate-lattice` references an
undefined helper, and `generate-positions` currently calls its lattice helper
with incompatible arguments. The `convert` utility expects a time-series
particle CSV with `iter` and `charge` columns, not the `type,x,y,z` input CSVs.
The `maze plot` group requires the optional plotting dependencies described in
the [README](../README.md).

For the first run, use:

```bash
maze run md examples/quickstart.yml
```

Inspect `examples/quickstart/output/energy.csv` and `temperature.csv` after
the run. The [paper examples](../examples/paper/README.md) show full NaCl and
water inputs.

Run from the repository root. Input file paths and output directories in the
configuration are interpreted relative to the current working directory.
