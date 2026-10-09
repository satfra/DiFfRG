# DiFfRG

**DiFfRG** (Discretization Framework for functional Renormalization Group flows)
is a C++20 scientific-computing library for solving fRG flow equations. It
provides spatial and temporal discretization of quantum field theory systems,
built on [deal.II](https://www.dealii.org/) (FEM),
[Kokkos](https://kokkos.org/) (GPU/CPU parallelization) and
[SUNDIALS](https://computing.llnl.gov/projects/sundials) (implicit solvers).

The framework spans three languages, each documented under its own tab above:

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} C++ Library
:link: doxygen/DiFfRG/html/index
:link-type: doc
The core library: models, discretizations, assemblers, timesteppers and
momentum-space integrators. Full API reference rendered by Doxygen.
:::

:::{grid-item-card} Python
:link: python
:link-type: doc
Post-processing tools: reading VTK/HDF5/CSV simulation output, plotting and
phase-diagram scans.
:::

:::{grid-item-card} Mathematica
:link: _generated_mathematica/index
:link-type: doc
Symbolic derivation of flow equations and generation of C++ integration
kernels.
:::

::::

New to DiFfRG? Start with the [installation guide](getting_started/installation.md)
and the [tutorials](tutorials/index.md).

```{toctree}
:hidden:
:maxdepth: 2

Getting Started <getting_started/index>
Tutorials <tutorials/index>
C++ API <doxygen/DiFfRG/html/index>
Python API <python>
Mathematica API <_generated_mathematica/index>
```
