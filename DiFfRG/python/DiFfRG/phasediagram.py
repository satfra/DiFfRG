import glob
import os
import shlex
import shutil
import subprocess

from DiFfRG.utilities import globalize
from DiFfRG.utilities import is_close
from DiFfRG.parallel import pmap, read_metas, sim_map
import DiFfRG.file_io as io

def get_all_sims(folder, pool_size=None, cache=True) -> list:
    """Reads every simulation in a folder.

    Only the config and the scalars are read here, in parallel; the fem frames
    of a run stay on disk until they are indexed (see `DiFfRG.file_io.hdf5`).
    A scan of long runs is mostly fem frames, and an analysis of it usually
    looks at the last frame of each or at none at all, so reading them all up
    front costs orders of magnitude more time and memory than the answer needs.

    Args:
        folder (str): The folder to read, with a trailing separator.
        pool_size (int, optional): The number of worker processes. Defaults to
            the number of cores this process may run on.
        cache (bool, optional): Whether to reuse what previous calls read of
            files that have not changed since. Defaults to True.

    Returns:
        list: One `SimulationData` per readable file. A file that cannot be read
        -- a run writing its output at this very moment, most likely -- is
        skipped and reported.
    """
    files_hdf5 = sorted(glob.glob(folder + "*.h5"))
    metas = read_metas(files_hdf5, workers=pool_size, cache=cache)
    if len(metas) != len(files_hdf5):
        print(f"{len(files_hdf5) - len(metas)} of {len(files_hdf5)} files in {folder} "
              "could not be read and are skipped")
    return [io.SimulationData(f, meta=metas[f]) for f in files_hdf5 if f in metas]

def sim_exists(sim_set, param_list) -> bool:
    for sim in sim_set:
        sim_config = sim.params
        match = True
        # param_list's names are path-like, so we need to split them by "/" and traverse the config dict accordingly
        for param in param_list:
            param_name = param[0].split("/")
            param_value = param[1]
            config_value = sim_config
            for name in param_name:
                if name in config_value:
                    config_value = config_value[name]
                else:
                    match = False
                    break
            if not match:
                break
            if is_close(config_value, param_value):
                match = False
                break
        if match:
            return True
    return False

def get_command(exe, param_list, folder="", add_params="", restart=None) -> str:
    if len(param_list) < 1:
        raise Exception(
            "When adding a point in the PD, you need to give as argument a list of [name, value] lists."
        )
    name = get_name(param_list)
    attach = ""
    for param in param_list:
        if len(param) != 2:
            raise Exception(
                "When adding a point in the PD, you need to give as argument a list of [name, value] lists."
            )
        attach = f"{attach} -sd {param[0]}={param[1]}"
    if restart is not None:
        add_params = f"--restart {shlex.quote(os.fspath(restart))} {add_params}"
    if folder != "":
        return f"{exe} -ss /output/name={name} {attach} -ss /output/folder={folder} {add_params}"
    return f"{exe} -ss /output/name={name} {attach} {add_params}"

def run_point(exe, param_list, add_params="", folder="", cwdir=os.getcwd(), suppress=True, restart=None) -> str:
    """Runs one point of a parameter scan.

    Args:
        restart (str, optional): A flow snapshot to continue from instead of starting at the initial
            scale, see `make_seed`. The run then takes its configuration from the snapshot, and
            `param_list` overrides it. Only meaningful if the parameters that differ from the seed run do
            not affect the flow above the snapshot's scale.
    """
    name = get_name(param_list)
    command = get_command(exe, param_list, folder, add_params, restart)
    if not suppress:
        print(f"Command: {command}")
    try:
        # os.system(get_command(exe, param_list, folder, add_params))
        if(suppress):
            subprocess.run(
                [command],
                shell=True,
                capture_output=False,
                text=False,
                cwd=cwdir,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL
            )
        else:
            subprocess.run(
                [command],
                shell=True,
                capture_output=False,
                text=False,
                cwd=cwdir
            )
    except:
        pass
    return name

def get_name(param_list) -> str:
    if len(param_list) < 1:
        raise Exception(
            "When adding a point in the PD, you need to give as argument a list of [name, value] lists."
        )
    name = ""
    for i, param in enumerate(param_list):
        if len(param) != 2:
            raise Exception(
                "When adding a point in the PD, you need to give as argument a list of [name, value] lists."
            )
        if i != 0:
            name = f"{name}_{param[0].split('/')[-1]}:{param[1]}"
        else:
            name = f"{param[0].split('/')[-1]}:{param[1]}"
    return name

def make_seed(exe, param_list, k_values, folder, add_params="", cwdir=os.getcwd(), suppress=True) -> list:
    """Runs a seed flow that writes snapshots at the scales `k_values` and stops after the last one.

    The typical use is a phase diagram: the effects of T and mu are suppressed as long as the RG
    scale is well above them (for fermions roughly like exp(-(k - mu) / T)), so one seed run, e.g.
    at low T and mu = 0, can be continued from a snapshot at such a k for every point of the scan:

        seed = pd.make_seed(exe, [["/physical/T", 0.005], ["/physical/muq", 0.0]], [2.0], "seeds/")
        pd.run_point(exe, [["/physical/T", 0.1], ["/physical/muq", 0.2]], folder="pd/",
                     restart=seed[0]["path"])

    Args:
        exe (str): The DiFfRG executable.
        param_list (list): [name, value] pairs, as for `run_point`.
        k_values (list): The RG scales of the snapshots; needs /physical/Lambda.
        folder (str): Where the seed run writes its output and snapshots.

    Returns:
        list: The `DiFfRG.file_io.snapshot_info` of every snapshot written, sorted by RG time.
    """
    ks = ",".join(repr(float(k)) for k in k_values)
    run_point(exe, param_list, add_params=f"--snapshots-k {ks} --stop-after-last-snapshot {add_params}",
              folder=folder, cwdir=cwdir, suppress=suppress)
    return io.list_snapshots(os.path.join(cwdir, folder, get_name(param_list)))
