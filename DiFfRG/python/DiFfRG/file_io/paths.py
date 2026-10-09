import os


def mkdir(path: str):
    """Creates a directory if it does not exist.

    Args:
        path (str): The path to the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_parameters_from_name(name: str) -> dict:
    """Extracts the parameters encoded in a DiFfRG output file name.

    Args:
        name (str): The file name, e.g. "sim_T:0.1_muq:0.3.pvd".

    Returns:
        dict: The parameter names and their values.
    """
    if name[-1] == "/":
        raise Exception("cannot read params from folder name")
    if name[-4:] == ".pvd" or name[-4:] == ".csv":
        raw_filename = name.split("/")[-1][:-4]
    else:
        raw_filename = name.split("/")[-1]

    params = {}
    splits = raw_filename.split("_")
    # merge a split with the next one if it does not contain a ":"
    i = 0
    while i < len(splits):
        if not ":" in splits[i]:
            if i == len(splits) - 1:
                splits.pop(i)
            else:
                splits[i] = splits[i] + "_" + splits[i + 1]
                splits.pop(i + 1)
        i += 1

    for p in splits:
        if len(p.split(":")) == 1:
            continue
        if not len(p.split(":")) == 2:
            raise Exception(
                "naming for file " + raw_filename + " could not be understood!"
            )
        param = p.split(":")[0]
        value = float(p.split(":")[1])
        params[param] = value
    return params
