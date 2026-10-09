import pandas
import numpy as np

def read_csv(csv, delim=",", header="infer"):
    """Reads a csv file and returns a pandas dataframe.

    Follows the same dialect as the C++ reader in DiFfRG/common/csv.hh and the
    Julia reader in julia/src/io/csv.jl: "#" comment lines and blank lines are
    skipped, cells that are not finite numbers read back as NaN, and rows whose
    field count does not match the header are skipped.

    Args:
        csv (str): The path to the csv file.
        delim (str, optional): Delimiter used in the csv file. Defaults to ",".
        header (str, optional): Header argument for pandas. Use None if no header exists. Defaults to "infer".

    Returns:
        pandas.DataFrame: The data in the csv file.
    """
    data = pandas.read_csv(
        csv,
        comment="#",
        delimiter=delim,
        header=header,
        skip_blank_lines=True,
        on_bad_lines="skip",
    )
    # Match the other readers, which collapse infinities into NaN so that every
    # gap in the data looks the same downstream.
    return data.replace([np.inf, -np.inf], np.nan)


def read_k_csv(filename, delim=",", kName="kGeV"):
    """Reads a csv file which contains data for different values of k and returns the data split into separate csvs for each value of k.

    Args:
        filename (str): The path to the csv file.
        delim (str, optional): Delimiter used in the csv file. Defaults to ",".
        kName (str, optional): The name of the column which contains the values of k. Defaults to "kGeV".

    Returns:
        tuple: A tuple containing a list with the unique values of k and a list of pandas dataframes for each value of k.

    """

    csv = read_csv(filename, delim=delim)
    # We need to split the data into separate csvs for each value of k
    ks = csv[kName]
    k_values = np.unique(ks)
    data = []
    for k in k_values:
        mask = ks == k
        data.append(csv[mask])
    return k_values, data

def split_csv(csv, name="kGeV"):
    """Reads a csv file which contains data for different values of k and returns the data split into separate csvs for each value of k.

    Args:
        filename (str): The path to the csv file.
        delim (str, optional): Delimiter used in the csv file. Defaults to ",".
        kName (str, optional): The name of the column which contains the values of k. Defaults to "kGeV".

    Returns:
        tuple: A tuple containing a list with the unique values of k and a list of pandas dataframes for each value of k.

    """

    vs = csv[name]
    v_values = np.unique(vs)
    data = []
    for v in v_values:
        mask = vs == v
        data.append(csv[mask])
    return v_values, data