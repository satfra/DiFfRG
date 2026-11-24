#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FEM/dg.hh>
#include <DiFfRG/discretization/FEM/ldg.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>

#include <limits>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace
{
  using namespace dealii;
  using namespace DiFfRG;

  template <typename Derived, typename Components_>
  class SpatialReadoutProbeBase : public def::AbstractModel<Derived, Components_>, public def::Time
  {
  public:
    using Components = Components_;

    template <typename Vector> void initial_condition(const Point<1> &x, Vector &values) const
    {
      values[0] = 2. * (x[0] - 0.2) + field_offset;
    }

    template <typename Vector> std::array<double, 1> EoM(const Point<1> &, const Vector &values) const
    {
      return {{values[0] - 0.6}};
    }

    template <typename NT, typename Solution>
    void extract(std::array<NT, 1> &extractors, const Point<1> &x, const Solution &sol) const
    {
      static_assert(Solution::size == 4 + Components::count_fe_subsystems());
      extractor_point = x[0];
      extractors[0] = get<"fe_functions">(sol)[0] + get<"fe_derivatives">(sol)[0][0];
    }

    template <typename DataOut, typename Solution>
    void capture_readout(const uint index, DataOut &, const Point<1> &x, const Solution &sol) const
    {
      static_assert(Solution::size == 4 + Components::count_fe_subsystems());
      readout_point[index] = x[0];
      extractor_point_at_readout[index] = extractor_point;
      field_value[index] = get<"fe_functions">(sol)[0];
      field_derivative[index] = get<"fe_derivatives">(sol)[0][0];
      readout_extractor[index] = get<"extractors">(sol)[0];
    }

    template <typename Helper, typename DataOut> void readouts_multiple(Helper &helper, DataOut &) const
    {
      helper(
          "left", [](const auto &, const auto &values) { return std::array<double, 1>{{values[0] - 0.2}}; },
          [&](auto &output, const auto &x, const auto &sol) { capture_readout(0, output, x, sol); });
      helper(
          "right", [](const auto &, const auto &values) { return std::array<double, 1>{{values[0] - 0.6}}; },
          [&](auto &output, const auto &x, const auto &sol) { capture_readout(1, output, x, sol); });
    }

    double field_offset = 0.;
    static constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    mutable std::array<double, 2> readout_point{{nan, nan}};
    mutable std::array<double, 2> extractor_point_at_readout{{nan, nan}};
    mutable double extractor_point = std::numeric_limits<double>::quiet_NaN();
    mutable std::array<double, 2> field_value{{nan, nan}};
    mutable std::array<double, 2> field_derivative{{nan, nan}};
    mutable std::array<double, 2> readout_extractor{{nan, nan}};
  };

  using ProbeComponents =
      ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<>, ExtractorDescriptor<Scalar<"e">>>;

  class SpatialReadoutProbe : public SpatialReadoutProbeBase<SpatialReadoutProbe, ProbeComponents>,
                              public def::LLFFlux<SpatialReadoutProbe>,
                              public def::FlowBoundaries<SpatialReadoutProbe>,
                              public def::FVDefaultBoundaries<SpatialReadoutProbe>,
                              public def::AD<SpatialReadoutProbe>
  {
  };

  using LDGProbeComponents = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<>,
                                                 ExtractorDescriptor<Scalar<"e">>, FEFunctionDescriptor<Scalar<"du">>>;

  class SpatialReadoutLDGProbe : public SpatialReadoutProbeBase<SpatialReadoutLDGProbe, LDGProbeComponents>,
                                 public def::NoNumFlux<SpatialReadoutLDGProbe>,
                                 public def::FlowBoundaries<SpatialReadoutLDGProbe>,
                                 public def::AD<SpatialReadoutLDGProbe>
  {
  public:
    template <uint dependent, int dim, typename NT, typename Solutions_s, typename Solutions_n>
    void ldg_numflux(std::array<Tensor<1, dim, NT>, 1> &, const Tensor<1, dim> &, const Point<dim> &,
                     const Solutions_s &, const Solutions_n &) const
    {
    }
  };

  ConfigTree make_json(const int fe_order)
  {
    return json::value({{"physical", {}},
                        {"discretization",
                         {{"fe_order", fe_order},
                          {"threads", 1},
                          {"overintegration", 0},
                          {"output_subdivisions", 1},
                          {"EoM_abs_tol", 1e-12},
                          {"EoM_max_iter", 100},
                          {"EoM_patch_order", 0},
                          {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:1:1"}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
                        {"output", {{"verbosity", 0}, {"vtk", false}, {"hdf5", false}}}});
  }

  template <typename Discretization, typename Assembler, typename Model>
  void check_spatial_readout(Model &model, const int fe_order, const bool origin_centered)
  {
    auto json = make_json(fe_order);
    if (origin_centered) model.field_offset = 1.2;
    RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json),
                                  RectangularMeshOptions{.origin_cell_centered = origin_centered}};
    Discretization discretization(mesh, json);
    Assembler assembler(discretization, model, json);
    FE::FlowingVariables state(discretization);
    state.interpolate(model);

    auto output_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "spatial_readout", "output");
    OutputSession<Discretization> output(output_path, json);
    output.write_frame(0., [&](auto &frame) {
      assembler.attach_data_output(frame, state.spatial_data(), typename Discretization::VectorType());
    });

    const std::array<double, 2> expected_point =
        origin_centered ? std::array<double, 2>{0., 0.} : std::array<double, 2>{.3, .5};
    const std::array<double, 2> expected_gradient =
        origin_centered ? std::array<double, 2>{.8, .8} : std::array<double, 2>{.2, .6};
    for (uint i = 0; i < 2; ++i) {
      CHECK(model.readout_point[i] == Catch::Approx(expected_point[i]).margin(1e-9));
      CHECK(model.extractor_point_at_readout[i] == Catch::Approx(model.readout_point[i]).margin(1e-12));
      CHECK(model.field_value[i] == Catch::Approx(expected_gradient[i]).margin(1e-9));
      if (!origin_centered) CHECK(model.field_derivative[i] == Catch::Approx(2.).margin(1e-8));
      CHECK(model.readout_extractor[i] == Catch::Approx(expected_gradient[i] + model.field_derivative[i]).margin(1e-8));
    }
  }
} // namespace

TEST_CASE("Spatial assemblers expose FE values and derivatives to readouts and extractors", "[discretization][readout]")
{
  try {
    spdlog::stdout_color_mt("log");
  } catch (const spdlog::spdlog_ex &) {
  }

  const bool origin_centered = GENERATE(false, true);

  SECTION("CG")
  {
    SpatialReadoutProbe model;
    using Discretization = CG::Discretization<SpatialReadoutProbe, RectangularMeshSerial<1>>;
    check_spatial_readout<Discretization, CG::Assembler<Discretization>>(model, 2, origin_centered);
  }
  SECTION("DG")
  {
    SpatialReadoutProbe model;
    using Discretization = DG::Discretization<SpatialReadoutProbe, RectangularMeshSerial<1>>;
    check_spatial_readout<Discretization, DG::Assembler<Discretization>>(model, 1, origin_centered);
  }
  SECTION("LDG")
  {
    SpatialReadoutLDGProbe model;
    using Discretization = LDG::Discretization<SpatialReadoutLDGProbe, RectangularMeshSerial<1>>;
    check_spatial_readout<Discretization, LDG::Assembler<Discretization>>(model, 1, origin_centered);
  }
  SECTION("KT-FV")
  {
    SpatialReadoutProbe model;
    using Discretization = FV::Discretization<SpatialReadoutProbe, RectangularMeshSerial<1>>;
    using Assembler = FV::KurganovTadmor::Assembler<Discretization>;
    check_spatial_readout<Discretization, Assembler>(model, 0, origin_centered);
  }
}

namespace
{
  template <typename Base> class EoMProbeAssembler : public Base
  {
  public:
    using Base::Base;
    template <typename Vector> auto locate(const Vector &solution) const
    {
      auto cell = this->get_discretization().get_dof_handler().begin_active();
      return this->locate_EoM(
          cell, solution, [](const auto &, const auto &u) { return std::array<double, 1>{u[0] - .6}; },
          [](const auto &p, const auto &) { return p; });
    }
  };
} // namespace

TEST_CASE("Assemblers select cell-average refinement only for FV data", "[discretization][EoM][FV][DG0]")
{
  try {
    spdlog::stdout_color_mt("log");
  } catch (const spdlog::spdlog_ex &) {
  }
  SpatialReadoutProbe model;
  auto config = make_json(0);
  config.set_uint("/discretization/EoM_patch_order", 3);
  RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(config)};
  const auto check = [&]<typename Discretization, typename Assembler>() {
    Discretization discretization(mesh, config);
    EoMProbeAssembler<Assembler> assembler(discretization, model, config);
    FE::FlowingVariables state(discretization);
    state.interpolate(model);
    return assembler.locate(state.spatial_data());
  };
  using DGSpace = DG::Discretization<SpatialReadoutProbe, RectangularMeshSerial<1>>;
  const auto dg = check.template operator()<DGSpace, DG::Assembler<DGSpace>>();
  CHECK_FALSE(dg.local_fit);
  CHECK(dg.point[0] == Catch::Approx(.5).margin(1e-10));
  using FVSpace = FV::Discretization<SpatialReadoutProbe, RectangularMeshSerial<1>>;
  const auto fv = check.template operator()<FVSpace, FV::KurganovTadmor::Assembler<FVSpace>>();
  REQUIRE(fv.local_fit != nullptr);
  CHECK(fv.point[0] == Catch::Approx(.5).margin(1e-10));
  CHECK((*fv.hessian)[0][0] == Catch::Approx(2.).margin(1e-10));
}
