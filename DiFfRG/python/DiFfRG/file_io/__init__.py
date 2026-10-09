r'''
Readers for DiFfRG simulation output.

The submodules are grouped by file format:

- `DiFfRG.file_io.csv`: csv output of e.g. flowing couplings
- `DiFfRG.file_io.hdf5`: hdf5 output, including the run configuration
- `DiFfRG.file_io.vtk`: pvd/vtu output of the field data
- `DiFfRG.file_io.paths`: helpers for the DiFfRG file naming scheme
- `DiFfRG.file_io.snapshots`: flow snapshots, from which a later run can continue

All of their public names are re-exported here, so `DiFfRG.file_io.SimulationData`
works just as well as `DiFfRG.file_io.hdf5.SimulationData`.
'''

import importlib

from DiFfRG.file_io import csv, hdf5, paths, snapshots
from DiFfRG.file_io.csv import read_csv, read_k_csv, split_csv
from DiFfRG.file_io.hdf5 import (LazyFEMData, SimulationData,
                                 get_config_from_hdf5, get_fem_frame_from_hdf5,
                                 get_fem_frames_from_hdf5, get_fem_from_hdf5,
                                 get_scalars_from_hdf5, read_meta)
from DiFfRG.file_io.paths import get_parameters_from_name, mkdir
from DiFfRG.file_io.snapshots import find_snapshot, list_snapshots, snapshot_info

# The vtk submodule is imported on first use only, so that the rest of the package
# stays usable when the (heavy, and often broken) vtk python bindings are unavailable.
__vtk_exports = ("PVDData", "PVDData1D", "SimulationData1D", "get_vtk_data")


def __getattr__(name):
    if name == "vtk" or name in __vtk_exports:
        module = importlib.import_module("DiFfRG.file_io.vtk")
        return module if name == "vtk" else getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + ["vtk"] + list(__vtk_exports))
