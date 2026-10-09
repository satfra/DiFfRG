#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FEM/ldg.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/timestepping/local_tolerances.hh>

#include <spdlog/sinks/stdout_color_sinks.h>

#include <cmath>

using namespace dealii;
using namespace DiFfRG;

namespace
{
  constexpr double abs_tol = 1e-10, rel_tol = 1e-6, refresh = 4.;

  /// The model's tolerance for component c at x: signed, so a negative u yields a non-positive tolerance.
  double model_tolerance(const uint c, const double x, const double u) { return (c + 1) * rel_tol * u * (1. + x); }

  template <bool with_hook>
  class ModelLDG
      : public def::AbstractModel<ModelLDG<with_hook>,
                                  ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<>,
                                                      ExtractorDescriptor<>, FEFunctionDescriptor<Scalar<"du">>>>,
        public def::Time,
        public def::NoNumFlux<ModelLDG<with_hook>>,
        public def::LDGUpDownFluxes<ModelLDG<with_hook>,
                                    def::UpDownFlux<def::FlowDirections<0>, def::UpDown<def::from_right>>>,
        public def::FlowBoundaries<ModelLDG<with_hook>>,
        public def::AD<ModelLDG<with_hook>>
  {
  public:
    ModelLDG() { this->components().add_dependency(1, 0, 0, 0); }

    template <typename Vector> void initial_condition(const Point<1> &x, Vector &values) const
    {
      values[0] = x[0] - 0.35;
    }

    template <uint submodel, typename NT, typename Vector>
    void ldg_flux(std::array<Tensor<1, 1, NT>, 1> &F, const Point<1> &, const Vector &u) const
    {
      F[0][0] = u[0];
    }

    template <int dim, typename Solution, size_t n>
    void abs_tolerances(std::array<double, n> &atol, const Point<dim> &x, const Solution &sol, double, double) const
      requires with_hook
    {
      atol[0] = model_tolerance(0, x[0], get<"fe_functions">(sol)[0]);
    }
  };

  class ModelKT
      : public def::AbstractModel<ModelKT, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>>>,
        public def::Time,
        public def::LLFFlux<ModelKT>,
        public def::FlowBoundaries<ModelKT>,
        public def::FVDefaultBoundaries<ModelKT>,
        public def::AD<ModelKT>
  {
  public:
    template <typename Vector> void initial_condition(const Point<1> &x, Vector &values) const
    {
      values[0] = x[0] - 0.35;
      values[1] = 0.5 + x[0];
    }

    template <int dim, typename Solution, size_t n>
    void abs_tolerances(std::array<double, n> &atol, const Point<dim> &x, const Solution &sol, double, double) const
    {
      for (uint c = 0; c < n; ++c)
        atol[c] = model_tolerance(c, x[0], get<"fe_functions">(sol)[c]);
    }
  };

  ConfigTree make_config(const int fe_order)
  {
    return ConfigTree(json::value({{"physical", {{"Lambda", 1.}}},
                                   {"discretization",
                                    {{"fe_order", fe_order},
                                     {"overintegration", 0},
                                     {"output_subdivisions", 1},
                                     {"EoM_abs_tol", 1e-10},
                                     {"EoM_max_iter", 0},
                                     {"grid", {{"x_grid", "0:0.1:1"}, {"refine", 0}}},
                                     {"adaptivity",
                                      {{"start_adapt_at", 0.},
                                       {"adapt_dt", 1e-1},
                                       {"level", 0},
                                       {"refine_percent", 1e-1},
                                       {"coarsen_percent", 5e-2}}}}},
                                   {"output", {{"live_plot", false}, {"verbosity", 0}}}}));
  }

  void ensure_logger()
  {
    try {
      spdlog::stdout_color_mt("log");
    } catch (const spdlog::spdlog_ex &) {
    }
  }

  /// The tolerance IDA must get for a model tolerance t: non-positive ones fall back to abs_tol.
  double expected(const double t) { return t > 0. ? t : abs_tol; }
} // namespace

TEST_CASE("LDG local tolerances sit at each dof's support point", "[timestepping][tolerances]")
{
  DiFfRG::Init();
  ensure_logger();
  using Model = ModelLDG<true>;
  using Discretization = LDG::Discretization<Model, RectangularMeshSerial<1>>;
  const auto config = make_config(GENERATE(1, 2));
  Model model;
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(config)};
  Discretization discretization(mesh, config);
  LDG::Assembler<Discretization> assembler(discretization, model, config);
  FE::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  auto u = state.spatial_data();

  std::vector<Point<1>> support(u.size());
  DoFTools::map_dofs_to_support_points(discretization.get_mapping(), discretization.get_dof_handler(), support);

  LocalAbsTolerances<typename Discretization::VectorType, typename Discretization::SparseMatrixType, 1> tol(
      assembler, abs_tol, rel_tol, refresh);
  REQUIRE(tol.init(u));
  bool negative_seen = false;
  for (uint i = 0; i < u.size(); ++i) {
    // FE_DGQ is nodal: the solution at a dof's support point is that dof's value.
    const double t = model_tolerance(0, support[i][0], u[i]);
    negative_seen |= t <= 0.;
    REQUIRE(tol.get()[i] == Catch::Approx(expected(t)).epsilon(1e-12));
  }
  REQUIRE(negative_seen);

  SECTION("refresh only on a tightening beyond the refresh factor")
  {
    const auto initial = tol.get();
    auto looser = u;
    looser *= 10.;
    REQUIRE_FALSE(tol.refresh_needed(looser));
    auto slightly_tighter = u;
    slightly_tighter *= 0.5;
    REQUIRE_FALSE(tol.refresh_needed(slightly_tighter));
    for (uint i = 0; i < u.size(); ++i)
      REQUIRE(tol.get()[i] == initial[i]);

    auto much_tighter = u;
    much_tighter *= 0.1;
    REQUIRE(tol.refresh_needed(much_tighter));
    for (uint i = 0; i < u.size(); ++i)
      REQUIRE(tol.get()[i] ==
              Catch::Approx(expected(model_tolerance(0, support[i][0], much_tighter[i]))).epsilon(1e-12));
  }
}

TEST_CASE("Without the model hook the stepper keeps its uniform tolerance", "[timestepping][tolerances]")
{
  DiFfRG::Init();
  ensure_logger();
  using Model = ModelLDG<false>;
  using Discretization = LDG::Discretization<Model, RectangularMeshSerial<1>>;
  const auto config = make_config(1);
  Model model;
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(config)};
  Discretization discretization(mesh, config);
  LDG::Assembler<Discretization> assembler(discretization, model, config);
  FE::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);

  LocalAbsTolerances<typename Discretization::VectorType, typename Discretization::SparseMatrixType, 1> tol(
      assembler, abs_tol, rel_tol, refresh);
  REQUIRE_FALSE(tol.init(state.spatial_data()));
  REQUIRE_FALSE(tol.refresh_needed(state.spatial_data()));
}

TEST_CASE("KT local tolerances sit at each cell centre, per component", "[timestepping][tolerances]")
{
  DiFfRG::Init();
  ensure_logger();
  // KT's local tolerances are serial-only (the reconstruction reads neighbour cells); a plain
  // RectangularMesh is partitioned in an MPI build.
  using Discretization = FV::Discretization<ModelKT, RectangularMeshSerial<1>, double>;
  const auto config = make_config(0);
  ModelKT model;
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(config)};
  Discretization discretization(mesh, config);
  FV::KurganovTadmor::Assembler<Discretization, ModelKT> assembler(discretization, model, config);
  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const auto &u = state.spatial_data();

  LocalAbsTolerances<typename Discretization::VectorType, typename Discretization::SparseMatrixType, 1> tol(
      assembler, abs_tol, rel_tol, refresh);
  REQUIRE(tol.init(u));
  const auto &fe = discretization.get_fe();
  std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
  for (const auto &cell : discretization.get_dof_handler().active_cell_iterators()) {
    cell->get_dof_indices(dofs);
    for (uint i = 0; i < dofs.size(); ++i) {
      const uint c = fe.system_to_component_index(i).first;
      // DG0: the reconstructed value at the cell centre is the cell value.
      REQUIRE(tol.get()[dofs[i]] ==
              Catch::Approx(expected(model_tolerance(c, cell->center()[0], u[dofs[i]]))).epsilon(1e-12));
    }
  }
}
