r'''
Readers for DiFfRG flow snapshots.

A run with `/timestepping/snapshots/k` (or `/t`) set writes one file per snapshot,
`<run name>_snapshot_<nnn>.h5`, next to its other output. Each holds the complete state of the
flow at one RG time, and a later run continues from it with `--restart <file>`.
'''

import glob
import json
import math
import os

import h5py


def snapshot_info(path) -> dict:
    """Reads the metadata of one snapshot file.

    Args:
        path (str): The snapshot file.

    Returns:
        dict: `path`, `t`, `k` (NaN if the run had no `/physical/Lambda`), `Lambda`, `dim`,
        `n_cells`, `n_variables`, `model_state` (the names of the stored model state entries)
        and `config`, the configuration of the run that wrote it, as a nested dictionary.
    """
    with h5py.File(path, "r") as f:
        config_json = f["config_json"][()]
        if isinstance(config_json, bytes):
            config_json = config_json.decode()
        mesh = f["mesh"]
        return {
            "path": os.fspath(path),
            "t": float(f.attrs["t"]),
            "k": float(f.attrs["k"]),
            "Lambda": float(f.attrs["Lambda"]),
            "dim": int(f.attrs["dim"]),
            "n_cells": len(mesh["active_cells"]) if "active_cells" in mesh else 0,
            "n_variables": int(f["state"].attrs["n_variables"]),
            "model_state": sorted(f["model"].keys()),
            "config": json.loads(config_json),
        }


def list_snapshots(run) -> list:
    """Lists the snapshots of a run, or of every run in a folder, ordered by RG time.

    Args:
        run (str): Either the path of a run without extension (e.g. `output/seed`, which finds
            `output/seed_snapshot_*.h5`) or a folder (which finds all `*_snapshot_*.h5` in it).

    Returns:
        list: One `snapshot_info` dictionary per snapshot, sorted by `t`.
    """
    # The run prefix is tried first: a run also creates a field directory named after itself, so
    # `output/seed` usually *is* a folder, just not the one holding the snapshots.
    paths = glob.glob(glob.escape(os.fspath(run)) + "_snapshot_*.h5")
    if not paths and os.path.isdir(run):
        paths = glob.glob(os.path.join(glob.escape(os.fspath(run)), "*_snapshot_*.h5"))
    infos = [snapshot_info(path) for path in paths]
    return sorted(infos, key=lambda info: info["t"])


def find_snapshot(run, k=None, t=None, rel_tol=1e-6) -> str:
    """Returns the snapshot file of a run at a given scale `k` or RG time `t`.

    Args:
        run (str): As for `list_snapshots`.
        k (float, optional): The RG scale of the snapshot.
        t (float, optional): The RG time of the snapshot. Exactly one of `k` and `t` must be given.
        rel_tol (float): Relative tolerance of the match. Snapshot times are moved onto the output
            grid by default, so a snapshot requested at `k` sits at the nearest output time, and a
            larger tolerance may be needed to find it by the requested value.

    Returns:
        str: The path of the matching snapshot.

    Raises:
        ValueError: if no snapshot, or more than one, matches.
    """
    if (k is None) == (t is None):
        raise ValueError("find_snapshot: give exactly one of k and t.")
    key, value = ("k", k) if k is not None else ("t", t)
    matches = [info for info in list_snapshots(run)
               if math.isclose(info[key], value, rel_tol=rel_tol, abs_tol=rel_tol)]
    if len(matches) != 1:
        raise ValueError(f"find_snapshot: {len(matches)} snapshots of '{run}' match {key} = {value}.")
    return matches[0]["path"]
