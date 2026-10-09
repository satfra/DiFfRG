# Python API

The DiFfRG Python package collects tools for analysing the results of DiFfRG
simulations:

- `DiFfRG.file_io` — read simulation output (VTK / HDF5 / CSV).
- `DiFfRG.plot` — plotting helpers built on matplotlib.
- `DiFfRG.phasediagram` — run parameter scans / phase-diagram sweeps from Python.
- `DiFfRG.utilities` — assorted helpers.

## Installation

The package ships as a wheel with every DiFfRG build:

```shell
pip install /path/to/DiFfRG/python/dist/DiFfRG-*.whl
```

The full API reference below is generated automatically from the package
docstrings.

```{toctree}
:maxdepth: 2

API Reference <autoapi/DiFfRG/index>
```
