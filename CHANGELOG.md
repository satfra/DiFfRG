# Changelog

## Version 1.1.0

### Changed

- Fully rebuilt build system as a CMake superbuild. Git submodules and the `external/` build scripts are gone; dependencies are now orchestrated by the top-level `CMakeLists.txt` via CPM/`ExternalProject`.
- Boost (>= 1.80), TBB (>= 2021) and SUNDIALS (>= 5.4.0) now use a compatible system copy automatically if found, otherwise build the bundled one. Controllable via `build.sh -b/-t/-s <prefix>` (or `-DBOOST_DIR`/`-DTBB_DIR`/`-DSUNDIALS_DIR`) and the `BUILD_BOOST`/`BUILD_TBB`/`BUILD_SUNDIALS` env vars.
- Compiler is selected via `CC`/`CXX`/`FC` and propagated to all dependencies.
- `build.sh` now auto-detects a sensible `-j` thread count and gains `-f` (full, non-interactive build).
- `doxygen-awesome-css` is vendored directly instead of being a submodule.

### Added

- Containers (Debian 13, Fedora 41, Rocky Linux 9, Ubuntu 24.04) and scripts to cross-test the build on other environments.

### Fixed

- Forgotten dependency on CPM source data caused by accidentally `INTERFACE` include directories.
- Attempted assignment to a `const` member inside a copy constructor.

## Version 1.0.1

### Fixed

- There was a bug with TaskFlow internally in deal.ii. Fixed for now by simply disabling taskflow in deal.ii.
- deal.ii changed its interface for dealii::SolutionTransfer. Adapted the corresponding methods.

### Changed

- The FlowingVariables classes are now in separate namespaces. For finite elements, use DiFfRG::FE::FlowingVariables, for pure variable systems use DiFfRG::FlowingVariables.
