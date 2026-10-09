r'''
Running an analysis over many simulations at once, and remembering the result.

A scan is a folder full of runs, and what is usually wanted from it is one number
per run: a condensate, the position of a shock, the outcome of a fit. Two things
make that slow, and this module addresses both.

- The work is done run by run in the notebook process. `sim_map` spreads it over
  a pool of workers instead. The function is shipped with cloudpickle, so a
  function defined in a notebook cell works, and only its (small) return value
  travels back.
- The same analysis is re-run every time a plot is redrawn. `sim_map` remembers
  what it computed for each file, in this process and in a cache file, so a
  second pass over an unchanged scan costs nothing.

The cache is keyed on the file (its path, its mtime and its size) *and* on the
code of the function: an edited analysis recomputes on its own, a rerun
simulation is picked up on its own, and neither needs the cache to be cleared by
hand. See `sim_map`.
'''

import atexit
import hashlib
import os
import pickle
import tempfile
from collections import OrderedDict

# --------------------------------------------------------------------- the pool

_executor_instance = None


def default_workers() -> int:
    """The number of worker processes used when none is given.

    The number of cores this process may actually run on, which under a batch
    system is the allocation rather than the size of the machine.
    """
    try:
        return min(32, len(os.sched_getaffinity(0)))
    except AttributeError:  # not on linux
        return min(32, os.cpu_count() or 1)


def get_executor(workers=None):
    """The process pool used by `pmap` and `sim_map`, created on first use.

    Deliberately not loky's `get_reusable_executor`: that one is a singleton
    shared with whatever else uses it -- an `adaptive` runner, most likely --
    and asking it for a different number of workers tears down its pool.

    Args:
        workers (int, optional): The number of worker processes. Defaults to
            `default_workers()`.

    Returns:
        loky.ProcessPoolExecutor: The pool.
    """
    global _executor_instance
    workers = default_workers() if workers is None else workers
    if _executor_instance is not None and _executor_instance._max_workers != workers:
        shutdown_executor()
    if _executor_instance is None:
        from loky import ProcessPoolExecutor

        # The workers exit after ten idle minutes, so an open notebook does not
        # sit on a node's worth of processes between two plots.
        _executor_instance = ProcessPoolExecutor(max_workers=workers, timeout=600)
        atexit.register(shutdown_executor)
    return _executor_instance


def shutdown_executor():
    """Stops the worker processes, if any are running."""
    global _executor_instance
    if _executor_instance is not None:
        _executor_instance.shutdown(wait=False, kill_workers=True)
        _executor_instance = None


def pmap(func, items, workers=None) -> list:
    """`[func(item) for item in items]`, evaluated over the process pool.

    Args:
        func (callable): What to apply. Shipped with cloudpickle, so a function
            defined in a notebook cell, a lambda or a closure all work.
        items (iterable): Its arguments, one per call.
        workers (int, optional): The number of worker processes. Defaults to
            `default_workers()`.

    Returns:
        list: The results, in the order of `items`.
    """
    items = list(items)
    if len(items) < 2:  # not worth a round trip
        return [func(item) for item in items]
    pool = get_executor(workers)
    chunksize = max(1, min(8, -(-len(items) // (4 * pool._max_workers))))
    return list(pool.map(func, items, chunksize=chunksize))


# -------------------------------------------------------------------- the cache

_cache_file = os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
    "DiFfRG", "sim_map.pkl",
)
_memory = {}
_disk = None
_disk_dirty = False

_meta_cache = OrderedDict()
_META_CACHE_SIZE = 4096


def cache_file(path=None) -> str:
    """The file `sim_map` stores its results in, optionally moving it.

    Defaults to `$XDG_CACHE_HOME/DiFfRG/sim_map.pkl`. The entries are keyed on
    the absolute path of the simulation file, so one cache serves every project.

    Args:
        path (str, optional): Where to keep the cache from now on. `None` leaves
            it where it is, `False` disables the on-disk cache for this session.

    Returns:
        str: The path in use.
    """
    global _cache_file, _disk, _disk_dirty
    if path is not None:
        _cache_file, _disk, _disk_dirty = path, None, False
    return _cache_file


def _load_disk_cache() -> dict:
    global _disk
    if _disk is None:
        try:
            with open(_cache_file, "rb") as f:
                _disk = pickle.load(f)
        except Exception:
            # A missing, truncated or unreadable cache is not an error: it only
            # means the work has to be done again.
            _disk = {}
    return _disk


def cache_save():
    """Writes the cache out, merging in whatever another session has added.

    Called at the end of every `sim_map`, so there is usually no reason to call
    it directly.
    """
    global _disk_dirty
    if not _disk_dirty or not _cache_file:
        return
    folder = os.path.dirname(_cache_file) or "."
    os.makedirs(folder, exist_ok=True)
    merged = dict(_load_disk_cache())
    try:
        with open(_cache_file, "rb") as f:
            merged.update(pickle.load(f))
    except Exception:
        pass
    merged.update(_disk)
    handle, temporary = tempfile.mkstemp(dir=folder)
    try:
        with os.fdopen(handle, "wb") as f:
            pickle.dump(merged, f, protocol=4)
        os.replace(temporary, _cache_file)  # atomic, so a reader never sees half a file
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    _disk_dirty = False


def cache_clear(disk=True):
    """Forgets every remembered result.

    Only needed when the cache is in the way -- an analysis whose result depends
    on something other than the simulation file and its own code, say. Editing
    the analysis or rerunning a simulation invalidates the affected entries by
    itself.

    Args:
        disk (bool, optional): Whether to delete the cache file as well.
    """
    global _disk, _disk_dirty
    _memory.clear()
    _meta_cache.clear()
    _disk, _disk_dirty = {}, False
    if disk and _cache_file and os.path.exists(_cache_file):
        os.remove(_cache_file)


def _file_key(path):
    """A file's identity: where it is, and what was last written to it."""
    stat = os.stat(path)
    return (os.path.realpath(path), stat.st_mtime_ns, stat.st_size)


def _code_key(func):
    """A digest of `func`, of the functions it calls, and of the constants it uses.

    cloudpickle serialises a function defined in a notebook by value and follows
    the globals it references, so a change anywhere in the analysis -- a
    threshold in a helper three calls down, say -- lands in this digest and the
    cached results for it are recomputed.
    """
    import cloudpickle

    return hashlib.sha1(cloudpickle.dumps(func)).hexdigest()[:16]


# ------------------------------------------------------------------- the mapping

def sim_map(func, sims, workers=None, cache=True) -> list:
    """`func(sim)` for every simulation, over the pool, remembered per file.

    Meant for the observables an analysis reduces a run to; the result has to be
    something small, since it travels back from a worker process and is kept in
    the cache. The whole point is that the run itself does not travel: a
    `SimulationData` is sent as the path it reads, and the worker reads the
    frames it needs.

    Args:
        func (callable): The analysis, applied to one `SimulationData`.
        sims (iterable): The simulations, e.g. from `get_all_sims`.
        workers (int, optional): The number of worker processes. Defaults to
            `default_workers()`.
        cache (bool, optional): Whether to remember the results. Defaults to True.

    Returns:
        list: `func`'s value for each simulation, in order.
    """
    global _disk_dirty
    sims = list(sims)
    if not cache:
        return pmap(func, sims, workers)

    disk = _load_disk_cache()
    code = _code_key(func)
    keys = [(code,) + _file_key(sim.hdf5_file) for sim in sims]
    missing = [i for i, key in enumerate(keys) if key not in _memory and key not in disk]
    if missing:
        for i, value in zip(missing, pmap(func, [sims[i] for i in missing], workers)):
            _memory[keys[i]] = disk[keys[i]] = value
        _disk_dirty = True
        cache_save()
    return [_memory[key] if key in _memory else disk[key] for key in keys]


def _safe_read_meta(path):
    """`read_meta`, or None if the file cannot be read.

    A simulation that is writing its output right now is the usual reason, and
    reading a folder should not fail because one run in it is still going.
    """
    from DiFfRG.file_io.hdf5 import read_meta

    try:
        return read_meta(path)
    except Exception:
        return None


def read_metas(paths, workers=None, cache=True) -> dict:
    """`read_meta` for many files at once, over the pool.

    Files that cannot be read are left out of the result rather than raising.
    What has been read before, and has not been written to since, is taken from
    memory: redrawing a plot then does not read the folder again. That memory is
    bounded, and holds the few thousand most recently used files.

    Args:
        paths (iterable): The hdf5 files to read.
        workers (int, optional): The number of worker processes.
        cache (bool, optional): Whether to reuse what was read earlier.

    Returns:
        dict: The path of every readable file, mapped to its `read_meta`.
    """
    paths = list(paths)
    keys, metas = {}, {}
    for path in paths:
        try:
            keys[path] = _file_key(path)
        except OSError:
            continue  # vanished between the glob and here
        if cache and keys[path] in _meta_cache:
            _meta_cache.move_to_end(keys[path])
            metas[path] = _meta_cache[keys[path]]

    missing = [path for path in keys if path not in metas]
    for path, meta in zip(missing, pmap(_safe_read_meta, missing, workers)):
        if meta is None:
            continue
        metas[path] = meta
        if cache:
            _meta_cache[keys[path]] = meta
    while len(_meta_cache) > _META_CACHE_SIZE:
        _meta_cache.popitem(last=False)
    return metas
