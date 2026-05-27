r'''
# The DiFfRG python package

The DiFfRG python package is a collection of tools for the analysis of results from simulations created with [DiFfRG](https://github.com/satfra/DiFfRG).

# Installation

To install the DiFfRG python package, you can either use the wheel file provided with a local DiFfRG build:

```shell
pip install ~/.local/share/DiFfRG/python/dist/DiFfRG-1.0.0-py3-none-any.whl
```

or you can download the wheel file from the DiFfRG repository and install it:

```shell
wget https://raw.githubusercontent.com/satfra/DiFfRG/main/python/dist/DiFfRG-1.0.0-py3-none-any.whl
pip install DiFfRG-1.0.0-py3-none-any.whl
```

# Usage

The DiFfRG python package provides a number of modules that can be used to analyze the results of DiFfRG simulations. The main modules are:

- `DiFfRG.file_io`: read DiFfRG simulation results from files
- `DiFfRG.plot`: plotting tools
- `DiFfRG.phasediagram`: tools to run DiFfRG simulations with different parameters directly from python

'''