import os

import h5py

def get_config_from_hdf5(file):
    """Reads the config from an hdf5 file and returns it as a nested dictionary.

    Args:
        file (str): The path to the hdf5 file."""
    
    # the config is stored at /config/ - groups are nested dictionaries with attributes.
    with h5py.File(file, "r") as f:
        config = {}
        # iterate over the groups in /config/
        for group in f["config"]:
            config[group] = {}
            for key, value in f["config"][group].attrs.items():
                config[group][key] = value
    return config

def get_fem_from_hdf5(file):
    """Reads the fem data from an hdf5 file and returns it as a nested dictionary.

    Args:
        file (str): The path to the hdf5 file."""
    
    # the fem data is stored at /FE/ - groups are nested dictionaries with attributes.
    with h5py.File(file, "r") as f:
        fem = []
        # the groups in /FE/ are named 000000, 000001, etc. - we need to sort them by name to get the correct order.
        for i, group in enumerate(sorted(f["FE"], key=lambda x: int(x))):
            fem.append({})
            for key, value in f["FE"][group].attrs.items():
                fem[-1][key] = value
            # Get all datasets in the group. We need to deduce their shapes (they may be 1D, 2D or 3D)
            for dataset in f["FE"][group]:
                fem[-1][dataset] = f["FE"][group][dataset][()]
        
    return fem

def get_scalars_from_hdf5(file):
    """Reads the scalars from an hdf5 file and returns them as a dictionary.

    Args:
        file (str): The path to the hdf5 file."""
    
    # the scalars are stored at /scalars/ - they're all stored as datasets
    with h5py.File(file, "r") as f:
        scalars = {}
        for key, value in f["scalars"].items():
            scalars[key] = value[()]
    return scalars


def get_fem_frames_from_hdf5(file):
    """Reads the names of the fem frames in an hdf5 file, in time order.

    Args:
        file (str): The path to the hdf5 file.

    Returns:
        list: The names of the groups below /FE/, sorted by their number.
    """
    with h5py.File(file, "r") as f:
        return sorted(f["FE"], key=lambda x: int(x)) if "FE" in f else []

def get_fem_frame_from_hdf5(file, frame):
    """Reads a single fem frame from an hdf5 file.

    Args:
        file (str): The path to the hdf5 file.
        frame (str): The name of the frame, as returned by `get_fem_frames_from_hdf5`.

    Returns:
        dict: The attributes and datasets of that frame.
    """
    with h5py.File(file, "r") as f:
        group = f["FE"][frame]
        data = dict(group.attrs)
        # Get all datasets in the group. We need to deduce their shapes (they may be 1D, 2D or 3D)
        for dataset in group:
            data[dataset] = group[dataset][()]
    return data

def read_meta(file):
    """Reads everything about a run except its fem data.

    This is what a `SimulationData` needs to be usable, and it is a tiny fraction
    of the file: a run writes one fem frame per output time, so the frames are
    typically three orders of magnitude larger than the rest.

    Args:
        file (str): The path to the hdf5 file.

    Returns:
        tuple: The config, the scalars and the names of the fem frames.
    """
    with h5py.File(file, "r") as f:
        config = {group: dict(f["config"][group].attrs) for group in f["config"]}
        scalars = {key: value[()] for key, value in f["scalars"].items()}
        frames = sorted(f["FE"], key=lambda x: int(x)) if "FE" in f else []
    return config, scalars, frames

class LazyFEMData:
    """The fem frames of a run, read from the file one at a time.

    Indexing, slicing, `len` and iteration work as they do on the list that
    `get_fem_from_hdf5` returns; the difference is that a frame is read when it
    is asked for, and kept from then on. Analyses that only look at the final
    state of a run -- most of them -- therefore touch a single frame instead of
    all of them.

    Args:
        file (str): The path to the hdf5 file.
        frames (list, optional): The frame names, if they are already known.
    """

    def __init__(self, file, frames=None):
        self.hdf5_file = file
        self.frames = get_fem_frames_from_hdf5(file) if frames is None else frames
        self._loaded = {}

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return (self[i] for i in range(len(self.frames)))

    def __repr__(self):
        return f"LazyFEMData({self.hdf5_file!r}, {len(self.frames)} frames)"

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self.frames)))]
        frame = self.frames[index]
        if frame not in self._loaded:
            self._loaded[frame] = get_fem_frame_from_hdf5(self.hdf5_file, frame)
        return self._loaded[frame]

    def load_all(self):
        """Reads every frame, and returns them as the plain list of dictionaries."""
        return [self[i] for i in range(len(self.frames))]

    def __reduce__(self):
        # Sent to another process as the file it reads, without the frames read
        # so far: the point of this class is not to ship arrays around.
        return (LazyFEMData, (self.hdf5_file, self.frames))

class SimulationData:
    """The output of one DiFfRG run: `params`, `scalars` and `fem_data`.

    The config and the scalars are read on construction, the fem frames when
    `fem_data` is indexed -- see `LazyFEMData`. Pass `lazy=False` for the old
    behaviour, in which `fem_data` is a plain list read up front.

    Args:
        name (str): The path to the hdf5 file.
        meta (tuple, optional): The result of `read_meta` for this file, if it
            has already been read -- `get_all_sims` reads it in a worker process
            and passes it in rather than reading the file twice.
        lazy (bool, optional): Whether to leave the fem frames on disk until they
            are used. Defaults to True.
    """

    def __init__(self, name, meta=None, lazy=True):
        self.hdf5_file = name

        if meta is None and not lazy:
            self.params = get_config_from_hdf5(self.hdf5_file)
            self.fem_data = get_fem_from_hdf5(self.hdf5_file)
            self.scalars = get_scalars_from_hdf5(self.hdf5_file)
            return

        self.params, self.scalars, frames = read_meta(self.hdf5_file) if meta is None else meta
        self.fem_data = LazyFEMData(self.hdf5_file, frames)
        if not lazy:
            self.fem_data = self.fem_data.load_all()

    def __repr__(self):
        return f"SimulationData({os.path.basename(self.hdf5_file)!r})"
