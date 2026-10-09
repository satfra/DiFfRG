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
  class RawPotentialProbeBase : public def::AbstractModel<Derived, Components_>, public def::Time
  {
  public:
    using Components = Components_;

    template <typename Vector> void initial_condition(const Point<1> &x, Vector &values) const
    {
      values[0] = 2. * (x[0] - 0.2);
    }

    template <typename Vector> std::array<double, 1> EoM(const Point<1> &, const Vector &values) const
    {
      return {{values[0] - 0.6}};
    }

    template <typename Vector>
    std::array<double, 1> raw_potential_gradient(const Point<1> &, const Vector &values) const
    {
      return {{values[0]}};
    }

    template <typename NT, typename Solution>
    void extract(std::array<NT, 1> &extractors, const Point<1> &x, const Solution &sol) const
    {
      extractor_point = x[0];
      extractor_potential = get<"potential">(sol);
      extractors[0] = get<"potential_gradient">(sol)[0] + get<"potential_hessian">(sol)[0][0];
    }

    template <typename DataOut, typename Solution>
    void capture_readout(const uint index, DataOut &, const Point<1> &x, const Solution &sol) const
    {
      readout_point[index] = x[0];
      extractor_point_at_readout[index] = extractor_point;
      extractor_potential_at_readout[index] = extractor_potential;
      potential[index] = get<"potential">(sol);
      potential_gradient[index] = get<"potential_gradient">(sol)[0];
      potential_hessian[index] = get<"potential_hessian">(sol)[0][0];
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

    static constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    mutable std::array<double, 2> readout_point{{nan, nan}};
    mutable std::array<double, 2> extractor_point_at_readout{{nan, nan}};
    mutable std::array<double, 2> extractor_potential_at_readout{{nan, nan}};
    mutable double extractor_point = std::numeric_limits<double>::quiet_NaN();
    mutable double extractor_potential = std::numeric_limits<double>::quiet_NaN();
    mutable std::array<double, 2> potential{{nan, nan}};
    mutable std::array<double, 2> potential_gradient{{nan, nan}};
    mutable std::array<double, 2> potential_hessian{{nan, nan}};
    mutable std::array<double, 2> readout_extractor{{nan, nan}};
  };

  using ProbeComponents =
      ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<>, ExtractorDescriptor<Scalar<"e">>>;

  class RawPotentialProbe : public RawPotentialProbeBase<RawPotentialProbe, ProbeComponents>,
                            public def::LLFFlux<RawPotentialProbe>,
                            public def::FlowBoundaries<RawPotentialProbe>,
                            public def::FVDefaultBoundaries<RawPotentialProbe>,
                            public def::AD<RawPotentialProbe>
  {
  };

  using LDGProbeComponents = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<>,
                                                 ExtractorDescriptor<Scalar<"e">>, FEFunctionDescriptor<Scalar<"du">>>;

  class RawPotentialLDGProbe : public RawPotentialProbeBase<RawPotentialLDGProbe, LDGProbeComponents>,
                               public def::NoNumFlux<RawPotentialLDGProbe>,
                               public def::FlowBoundaries<RawPotentialLDGProbe>,
                               public def::AD<RawPotentialLDGProbe>
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
                          {"EoM_smoothing_length", -1.},
                          {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:1:1"}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
                        {"output", {{"verbosity", 0}, {"vtk", false}, {"hdf5", false}}}});
  }

  template <typename Discretization, typename Assembler, typename Model>
  void check_raw_potential_readout(Model &model, const int fe_order)
  {
    auto json = make_json(fe_order);
    RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
    Discretization discretization(mesh, json);
    Assembler assembler(discretization, model, json);
    FE::FlowingVariables state(discretization);
    state.interpolate(model);

    auto output_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "raw_potential", "output");
    OutputSession<Discretization> output(output_path, json);
    output.write_frame(0., [&](auto &frame) {
      assembler.attach_data_output(frame, state.spatial_data(), typename Discretization::VectorType());
    });

    const std::array<double, 2> expected_point{{0.3, 0.5}};
    const std::array<double, 2> expected_potential{{-0.03, 0.05}};
    const std::array<double, 2> expected_gradient{{0.2, 0.6}};
    for (uint i = 0; i < 2; ++i) {
      CHECK(model.readout_point[i] == Catch::Approx(expected_point[i]).margin(5e-2));
      CHECK(model.extractor_point_at_readout[i] == Catch::Approx(model.readout_point[i]).margin(1e-12));
      CHECK(model.extractor_potential_at_readout[i] == Catch::Approx(model.potential[i]).margin(1e-12));
      CHECK(model.potential[i] == Catch::Approx(expected_potential[i]).margin(5e-2));
      CHECK(model.potential_gradient[i] == Catch::Approx(expected_gradient[i]).margin(5e-2));
      CHECK(model.potential_hessian[i] == Catch::Approx(2.).margin(1e-1));
      CHECK(model.readout_extractor[i] == Catch::Approx(expected_gradient[i] + 2.).margin(1e-1));
    }
  }
} // namespace

TEST_CASE("Spatial assemblers expose one raw-potential view to readouts and extractors",
          "[discretization][raw-potential][readout]")
{
  try {
    spdlog::stdout_color_mt("log");
  } catch (const spdlog::spdlog_ex &) {
  }

  SECTION("CG")
  {
    RawPotentialProbe model;
    using Discretization = CG::Discretization<RawPotentialProbe, RectangularMeshSerial<1>>;
    check_raw_potential_readout<Discretization, CG::Assembler<Discretization>>(model, 2);
  }
  SECTION("DG")
  {
    RawPotentialProbe model;
    using Discretization = DG::Discretization<RawPotentialProbe, RectangularMeshSerial<1>>;
    check_raw_potential_readout<Discretization, DG::Assembler<Discretization>>(model, 1);
  }
  SECTION("LDG")
  {
    RawPotentialLDGProbe model;
    using Discretization = LDG::Discretization<RawPotentialLDGProbe, RectangularMeshSerial<1>>;
    check_raw_potential_readout<Discretization, LDG::Assembler<Discretization>>(model, 1);
  }
  SECTION("KT-FV")
  {
    RawPotentialProbe model;
    using Discretization = FV::Discretization<RawPotentialProbe, RectangularMeshSerial<1>>;
    using Assembler = FV::KurganovTadmor::Assembler<Discretization>;
    check_raw_potential_readout<Discretization, Assembler>(model, 0);
  }
}
