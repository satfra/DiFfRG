import glob
import json
import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np
import pandas
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from DiFfRG.utilities import globalize

# A class to read in .pvd files
class PVDData:
    def __read_pvd(self, only_one: bool = False, at_t: int = -1, pool_size: int = 32):
        @globalize
        def load_one_step(i):
            file = self.files[i]
            t = self.timesteps[i]

            # check if the data is already loaded
            if t in self.slice_cache:
                return self.slice_cache[t]

            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(file)
            reader.Update()
            polydata = reader.GetOutput()
            nodes, data = get_vtk_data(polydata)

            # check if there are any NaNs in the data
            for sub_data in data.values():
                if np.any(np.isnan(sub_data)):
                    print("NaNs in data at t = ", t)

                    # remove this timestep from the cache
                    m_idx = int(np.argmin(np.abs(self.timesteps - t)))
                    if np.isclose(self.timesteps[m_idx], t):
                        self.timesteps = np.delete(self.timesteps, m_idx)
                        self.files.pop(m_idx)

                        new_idx = int(np.argmin(np.abs(self.timesteps - t)))
                        return load_one_step(new_idx)

            self.slice_cache[t] = {"t": t, "nodes": nodes, "point_data": data}

            return self.slice_cache[t]

        if only_one:
            if at_t < 0:
                idx = -1
            else:
                idx = np.argmin(np.abs(self.timesteps - at_t))
            return load_one_step(idx)

        pool = Pool(pool_size)
        return pool.map(load_one_step, range(len(self.files)))

    def __init__(self, filename : str):
        self.filename = filename
        self.slice_cache = {}

        if filename[-4:] != ".pvd":
            raise Exception("only .pvd files are supported!")

        self.dir = os.path.dirname(os.path.abspath(filename))
        if self.dir[-1] != "/":
            self.dir = self.dir + "/"

        self.timesteps = []
        self.files = []
        tree = ET.parse(filename)
        root = tree.getroot()
        for timestep_data in root.iter("DataSet"):
            self.timesteps.append(float(timestep_data.get("timestep")))
            self.files.append(self.dir + timestep_data.get("file"))
        self.timesteps = np.array(self.timesteps)

    def get_full_data(self):
        if not self.data:
            self.data = PVDData.__read_pvd(self.filename)
        return self.data

    def get_data_slice(self, t : float = -1):
        if t < 0:
            t = self.timesteps[-1]
        t = self.timesteps[np.argmin(np.abs(self.timesteps - t))]
        if t in self.slice_cache:
            return self.slice_cache[t]
        self.slice_cache[t] = self.__read_pvd(only_one=True, at_t=t)
        return self.slice_cache[t]

class PVDData1D(PVDData):
    def __init__(self, filename : str, cs=0, mass_name="u"):
        super().__init__(filename)

    def get_raw_x(self, t : float = -1):
        return self.get_data_slice(t)["nodes"][:, 0]
    
    def get_raw_array(self, name : str, t : float = -1):
        return self.get_data_slice(t)["point_data"][name]
    
    def get_x(self, t : float = -1):
        x = self.get_raw_x(t)
        # find all duplicates
        duplicates = np.where(np.diff(x) == 0)[0]
        # merge duplicates in x and average in y
        x = np.delete(x, duplicates)
        return x
    
    def get_array(self, name : str, t : float = -1):
        x = self.get_raw_x(t)
        y = self.get_raw_array(name, t)
        # find all duplicates
        duplicates = np.where(np.diff(x) == 0)[0]
        # merge duplicates in x and average in y
        x = np.delete(x, duplicates)
        # average in y
        y[duplicates] = (y[duplicates] + y[duplicates + 1]) / 2
        y = np.delete(y, duplicates + 1)
        return y
    
class SimulationData1D(PVDData1D):
    def __init__(self, name):
        self.pvd_file = name + ".pvd"
        super().__init__(self.pvd_file)

        # Only written when there is no HDF5 output to carry the configuration, or when
        # /output/json is set. A run with HDF5 on keeps it in the .h5 file's /config group.
        log_json = name + ".log.json"
        self.params = json.load(open(log_json)) if os.path.exists(log_json) else None

        # find all associated csv files
        self.csv_files = glob.glob(name + "_*.csv")
        # load them using pandas
        self.csv_data = {}
        for f in self.csv_files:
            self.csv_data[f] = pandas.read_csv(f)

    def get_csv(self, name):
        for f in self.csv_files:
            if name in f:
                return self.csv_data[f]
        return None


def get_vtk_data(vtkdata):
    """Utility function to extract nodes and data from a (loaded) vtk file.

    Args:
        vtkdata : The loaded vtk data.

    Returns:
        tuple: A tuple containing a numpy array of nodes and a dict with the arrays in the vtkdata.
    """
    
    nodes = vtk_to_numpy(vtkdata.GetPoints().GetData())
    data = {}

    number_of_arrays = vtkdata.GetPointData().GetNumberOfArrays()
    for i in range(number_of_arrays):
        name = vtkdata.GetPointData().GetArray(i).GetName()
        data[name] = vtk_to_numpy(vtkdata.GetPointData().GetAbstractArray(name))

    return nodes, data