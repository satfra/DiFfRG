# Project Structure

## Core Pipeline

DiFfRG follows a **Model → Discretization → Assembler → TimeStepper → Output** pipeline:

1. **Model** — The user defines the physics by inheriting from `def::AbstractModel<Model, Components>`. A model specifies the mass function, flux, source terms, and optionally numerical fluxes and boundary conditions; see [Numerical models](models.md).

2. **Components** — A descriptor type that declares how many FE functions (and their polynomial orders), extractors, and additional variables the model uses.

3. **Discretization / Assembler** — An assembler takes the model and a mesh, then computes residuals and Jacobians using a chosen spatial discretization:
   - **CG** (Continuous Galerkin)
   - **DG** (Discontinuous Galerkin)
   - **LDG** (Local Discontinuous Galerkin, for higher-order derivatives)
   - **FV** (Finite Volume, Kurganov-Tadmor scheme)

4. **TimeStepper** — Evolves the discretized system in RG time. Options include:
   - *Explicit*: Forward Euler, Boost Runge-Kutta, Adams-Bashforth-Moulton
   - *Implicit*: SUNDIALS IDA, implicit Euler, TRBDF2
   - The linear solver of the implicit steppers (UMFPack, GMRES, ...) is a template argument; see [Timesteppers](timesteppers.md).

5. **Output** — An `OutputSession` writes the solution and the model's readouts to HDF5, VTK and CSV files at regular RG-time intervals; see [Output](output.md).

## Directory Layout

```
DiFfRG/
├── include/DiFfRG/
│   ├── common/          Math utilities, JSON/TOML config, quadrature, Kokkos wrappers
│   ├── discretization/  Assemblers (CG/DG/LDG/FV), mesh, coordinates, data output
│   ├── model/           AbstractModel (CRTP), component descriptors, numerical fluxes
│   ├── physics/         Integrators, interpolators, regulators, threshold functions
│   └── timestepping/    Explicit/implicit steppers, linear/nonlinear solvers
├── src/DiFfRG/          Corresponding .cc source files
├── tests/               Catch2 test sources
├── python/              Post-processing (VTK/HDF5 I/O, plotting)
├── Mathematica/         Symbolic computation and C++ code generation
└── documentation/       This documentation (Sphinx + Doxygen)
```

## Configuration

Simulations are configured via a `parameter.json` or `parameter.toml` file, which is read by `ConfigurationHelper` into a `ConfigTree`. Key sections (all keys are listed in the [parameter reference](parameters.md)):

- `/physical/` — Physics parameters (temperature, couplings, etc.)
- `/integration/` — Quadrature orders and tolerances for momentum integrals
- `/discretization/` — FE order, grid specification, adaptivity settings
- `/timestepping/` — Final time, output interval, explicit/implicit solver tolerances
- `/output/` — Output folder, name, verbosity

CLI flags (`-sd`, `-si`, `-sb`, `-ss`) can override any parameter at runtime.

### File formats

Both JSON and TOML are accepted; the format is chosen from the file extension, `.toml` (or `.tml`) being read as TOML and everything else as JSON. TOML allows comments and needs less punctuation, which makes it the friendlier choice for hand-written parameter files:

```toml
[physical]
Lambda = 1.0
T = 0.1        # in units of Lambda

[discretization.grid]
x_grid = "0:1e-2:1"
```

Nothing else changes with the format. Parameters are always addressed by the same path, both from C++ (`config.get_double("/physical/T")`) and on the command line (`-sd /physical/T=0.1`), and the configuration log written next to the output is always JSON.

If no parameter file is given with `-p` and the default `parameter.json` does not exist, but `parameter.toml` does, the latter is used. A file named explicitly with `-p` is never substituted. `--generate-parameter-file` (or `--gen-toml`) writes a TOML file with the default values, `--gen-json` a JSON file; the file is named like the parameter file and an existing file is never replaced.

## Key Design Patterns

- **CRTP** (Curiously Recurring Template Pattern): a model passes its own type to its base class, so the library calls the model's methods without virtual-function overhead.
- **Kokkos execution spaces** (`GPU_exec`, `TBB_exec`, `KokkosHost_exec`) abstract the CPU/GPU backend.
- **C++20 concepts** constrain template parameters with clear diagnostics.
- **Automatic differentiation** computes Jacobians without hand-coded derivatives.

## Parallelization (Kokkos)

DiFfRG uses [Kokkos](https://kokkos.org/) as a performance-portability layer, so the same kernel code runs on different backends selected through *execution spaces*:

- `TBB_exec` — CPU parallelization via [oneTBB](https://github.com/uxlfoundation/oneTBB).
- `KokkosHost_exec` — the Kokkos host backend. This is `Kokkos::Serial` by default: DiFfRG's CPU work runs on the single TBB pool above, and a second pool of spinning workers on the same cores would only contend with it. Building with `-DKOKKOS_THREADS=ON` makes it Kokkos' native C++ threads backend instead.
- `GPU_exec` — GPU execution via Kokkos' CUDA (NVIDIA) or HIP (AMD) backend.

The GPU backend is enabled at configure time with `-DGPU=ON` (the default; see the [installation](installation.md) page). Whether a particular computation runs on the CPU or the GPU is chosen per execution space: the **FEM assemblers always run on the CPU (`TBB_exec`)**, because they interact with deal.II's data structures, while the **momentum-space integrators can target the GPU** for a large speedup on fully momentum-dependent flows. The integrator's execution space is fixed when its kernel is generated (the `"Device"` option of `MakeKernel`, see [Tutorial 3](../tutorials/tut3.md)); `TBB` is the default and is required when an integrator is called from within an assembler.

Because the backend is abstracted, models and generated kernels are written once (using the `KOKKOS_INLINE_FUNCTION` markers in the generated code) and compile for whichever backend is enabled.
