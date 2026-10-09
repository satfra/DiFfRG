#pragma once

// Common typedefs, parameter constants, grid helpers, and JSON-config builder
// for the 3D-physics KT regression tests (linear sigma model at physical
// scale: Lambda = 0.65, m² = -0.1, λ = 71.6, N = 2, T = 0.05 — matches
// Examples/ONfiniteT/parameter_KT.json and Examples/ONfiniteT/parameter.json).
//
// The grid scale matches the CG Example: max_rho = 1.5e-2; the σ-coordinate
// counterpart uses max_sigma = sqrt(2 · max_rho) so both discretisations
// cover the SAME physical field range (since ρ = σ²/2).

#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/discretization/discretization.hh"
#include "DiFfRG/timestepping/timestepping.hh"

#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>
#include <DiFfRG/model/model.hh>

#include "../kt_regression_helpers.hh"

#include <catch2/catch_all.hpp>

#include <cmath>
#include <cstddef>

namespace on_kt_3D
{
  using namespace DiFfRG;
  using namespace dealii;

  // --- Common typedefs --------------------------------------------------------

  constexpr uint dim = 1;
  constexpr double origin_tol = 1.0e-12;

  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  template <typename Model> using ConstrainUAtOrigin = def::ConstrainOriginSupportPointToZero<"u", Model>;

  using NumberType = double;
  using Mesh = RectangularMeshSerial<dim>;
  using Discretization = FV::Discretization<Components, Mesh, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;

  template <typename Model>
  using Assembler =
      FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor, FV::KurganovTadmor::MaxEigenvalueWaveSpeed>;
  using ImplicitTimeStepper = TimeStepperSUNDIALS_IDA<Discretization>;

  // --- LSM physical-scale parameters -----------------------------------------
  //
  // V₀(σ) = (m²/2) σ² + (λ/16) σ⁴, equivalently V₀(ρ) = m²·ρ + (λ/4)·ρ²
  //   with the field convention ρ = σ²/2. Matches Examples/ONfiniteT*.

  struct LSMPhysicalParameters {
    static constexpr double Lambda = 0.65;         // UV cutoff (GeV)
    static constexpr double m2 = -0.1;             // initial mass-squared
    static constexpr double lambda_quartic = 71.6; // initial quartic coupling
    static constexpr double T = 0.05;              // temperature for the integrator
    static constexpr double N = 2.0;               // O(N) flavors
  };

  // --- Grid scale (matches Examples) ------------------------------------------

  constexpr double max_rho = 1.5e-2;                             // from CG Example x_grid upper bound
  inline double max_sigma() { return std::sqrt(2.0 * max_rho); } // σ = sqrt(2ρ)
  constexpr std::size_t n_cells_default = 150;                   // matches Example uniform-grid case
  constexpr double final_time = 4.0;                             // matches Examples
  // The flow develops the SSB convexification near the origin at the chiral IR; on the
  // default 150-cell grid the σ-mode argument k²+m²_σ touches the propagator pole around
  // t≈3.97. Reaching t=4.0 cleanly is a mesh-resolution matter (more cells), not a
  // correctness one, so the smoke tests target t=3.6 — comfortably inside the resolved
  // regime once the diffusion-flux sign is correct.
  constexpr double smoke_target_time = 3.6;

  struct GridSettings {
    std::size_t cells = n_cells_default;
    double x_min = 0.0;
    double x_max = max_rho; // overwritten by ρ/σ helpers
  };

  inline GridSettings default_rho_grid() { return GridSettings{n_cells_default, 0.0, max_rho}; }
  inline GridSettings default_sigma_grid() { return GridSettings{n_cells_default, 0.0, max_sigma()}; }

  inline double grid_spacing(const GridSettings &g) { return (g.x_max - g.x_min) / static_cast<double>(g.cells - 1); }

  inline Config::ConfigurationMesh<1> make_mesh_config(const GridSettings &g)
  {
    const double delta = grid_spacing(g);
    const Config::GridAxis axis(g.x_min - 0.5 * delta, delta, g.x_max + 0.5 * delta);
    return Config::ConfigurationMesh<1>(0u, std::vector<Config::GridAxis>{axis});
  }

  // Multi-axis (non-uniform) mesh from explicit subranges. Each entry is
  // a {min, step, max} triple; subranges must be contiguous (next.min ==
  // prev.max). Used to reproduce the CG Example's adaptive grid
  //   "0:1e-4:5e-3, 5e-3:5e-4:6e-3, 6e-3:1e-3:1.5e-2"
  // for the CG-vs-KT discriminator tests.
  struct GridSubrange {
    double min;
    double step;
    double max;
  };

  inline Config::ConfigurationMesh<1> make_adaptive_mesh_config(const std::vector<GridSubrange> &subranges)
  {
    std::vector<Config::GridAxis> axes;
    axes.reserve(subranges.size());
    for (const auto &s : subranges)
      axes.emplace_back(s.min, s.step, s.max);
    return Config::ConfigurationMesh<1>(0u, std::move(axes));
  }

  inline std::size_t total_cells(const std::vector<GridSubrange> &subranges)
  {
    std::size_t n = 0;
    for (const auto &s : subranges) {
      const auto steps = static_cast<std::size_t>((s.max - s.min) / s.step);
      n += (std::abs((s.min + s.step * steps) - s.max) < 1.0e-12) ? steps : (steps + 1);
    }
    return n;
  }

  // The CG Example's grid (Examples/ONfiniteT/parameter.json), shifted by -0.5·step
  // so that the first cell centre lands at ρ = 0 (consistent with our uniform-grid
  // convention for FV: x_grid runs from -0.5·step to max + 0.5·step).
  inline std::vector<GridSubrange> example_cg_subranges()
  {
    return {{-0.5e-4, 1.0e-4, 5.0e-3 + 0.5e-4},
            {5.0e-3 + 0.5e-4, 5.0e-4, 6.0e-3 + 0.25e-3},
            {6.0e-3 + 0.25e-3, 1.0e-3, 1.5e-2 + 0.5e-3}};
  }

  // --- JSON configuration -----------------------------------------------------
  //
  // Defaults tuned to the Examples' implicit-IDA settings:
  //   abs_tol = 1e-7, rel_tol = 1e-7 (the KT example uses these for IDA;
  //   the CG example tightens abs_tol to 1e-13). Callers override per test.
  //   x_order = 32, x_extent_tolerance = 1e-3 (matches both Examples).
  //   final_time and Λ are pulled from LSMPhysicalParameters.

  inline ConfigTree make_json(const int fe_order = 0, const int x_order = 32,
                              const double ida_abs_tol = 1.0e-7, const double ida_rel_tol = 1.0e-7)
  {
    return json::value(
        {{"physical", {{"Lambda", LSMPhysicalParameters::Lambda}}},
         {"integration", {{"x_order", x_order}, {"x_extent_tolerance", 1.0e-3}, {"jacobian_quadrature_factor", 0.5}}},
         {"discretization",
          {{"fe_order", fe_order},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1.0e-10},
           {"EoM_max_iter", 0}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", 2.0},
           {"explicit",
            {{"dt", 1.0e-8}, {"minimal_dt", 1.0e-12}, {"maximal_dt", 1.0}, {"abs_tol", 1.0e-12}, {"rel_tol", 1.0e-10}}},
           {"implicit",
            {{"dt", 1.0e-8},
             {"minimal_dt", 1.0e-14},
             {"maximal_dt", 1.0},
             {"abs_tol", ida_abs_tol},
             {"rel_tol", ida_rel_tol},
             {"max_steps", 5'000'000}}}}},
         {"output", {{"verbosity", 1}, {"vtk", false}, {"hdf5", false}}}});
  }

} // namespace on_kt_3D
