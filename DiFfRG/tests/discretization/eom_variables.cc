#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>

#include <limits>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <type_traits>

// A model whose EoM depends on the variables defines EoM(x, u, variables). The assembler must hand it the variables
// of the state it is working on, both when it extracts and when it writes the readouts, so that the EoM follows them
// immediately and never lags behind a previous call.

namespace
{
  using namespace dealii;
  using namespace DiFfRG;

  using Components = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<Scalar<"c">>,
                                         ExtractorDescriptor<Scalar<"e">>>;

  // u(x) = 2 (x - 0.2) and EoM u = c, so the EoM sits at x = 0.2 + c / 2.
  class Probe : public def::AbstractModel<Probe, Components>,
                public def::Time,
                public def::LLFFlux<Probe>,
                public def::FlowBoundaries<Probe>,
                public def::FVDefaultBoundaries<Probe>,
                public def::AD<Probe>
  {
  public:
    template <typename Vector> void initial_condition(const Point<1> &x, Vector &values) const
    {
      values[0] = 2. * (x[0] - 0.2);
    }

    template <typename Vector> void initial_condition_variables(Vector &values) const { values[0] = 0.6; }

    template <typename Vector, typename Variables>
    std::array<double, 1> EoM(const Point<1> &, const Vector &u, const Variables &variables) const
    {
      return {{u[0] - variables[0]}};
    }

    template <typename Vector>
    std::array<double, 1> raw_potential_gradient(const Point<1> &, const Vector &values) const
    {
      return {{values[0]}};
    }

    template <typename NT, typename Solution>
    void extract(std::array<NT, 1> &extractors, const Point<1> &x, const Solution &sol) const
    {
      extractors[0] = get<"fe_functions">(sol)[0];
      if constexpr (std::is_same_v<NT, double>) extract_point = x[0];
    }

    template <int d, typename DataOut, typename Solution>
    void readouts(DataOut &, const Point<d> &x, const Solution &) const
    {
      readout_point = x[0];
    }

    static constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    mutable double extract_point = nan;
    mutable double readout_point = nan;
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
                          {"grid", {{"x_grid", "0:0.02:1"}, {"y_grid", "0:1:1"}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
                        {"output", {{"verbosity", 0}, {"vtk", false}, {"hdf5", false}}}});
  }

  double EoM_at(const double c) { return 0.2 + c / 2.; }

  template <typename Discretization, typename Assembler> void check(const int fe_order, const double tol)
  {
    Probe model;
    auto json = make_json(fe_order);
    RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
    Discretization discretization(mesh, json);
    Assembler assembler(discretization, model, json);
    FE::FlowingVariables state(discretization);
    state.interpolate(model);
    auto &c = state.variable_data()[0];

    auto output_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "eom_variables", "output");
    OutputSession<Discretization> output(output_path, json);
    double time = 0.;
    const auto write_frame = [&] {
      output.write_frame(
          time, [&](auto &frame) { assembler.attach_data_output(frame, state.spatial_data(), state.variable_data()); });
      time += 1.;
    };

    // The readouts and the extractors in a frame both sit at the EoM of the variables written out.
    write_frame();
    CHECK(model.readout_point == Catch::Approx(EoM_at(0.6)).margin(tol));
    CHECK(model.extract_point == Catch::Approx(EoM_at(0.6)).margin(tol));

    // Changing the variables moves the EoM in the very next frame.
    c = 0.2;
    write_frame();
    CHECK(model.readout_point == Catch::Approx(EoM_at(0.2)).margin(tol));
    CHECK(model.extract_point == Catch::Approx(EoM_at(0.2)).margin(tol));

    // The same for the extractors of the variables' residual.
    c = 0.4;
    typename Discretization::VectorType residual(state.variable_data());
    assembler.residual_variables(residual, state.variable_data(), state.spatial_data());
    CHECK(model.extract_point == Catch::Approx(EoM_at(0.4)).margin(tol));
  }
} // namespace

TEST_CASE("An EoM that reads the variables sees the current ones", "[discretization][eom]")
{
  try {
    spdlog::stdout_color_mt("log");
  } catch (const spdlog::spdlog_ex &) {
  }

  SECTION("CG")
  {
    using D = CG::Discretization<Probe, RectangularMeshSerial<1>>;
    check<D, CG::Assembler<D>>(2, 5e-3);
  }
  // FV is piecewise constant, so the EoM is only resolved to a cell width.
  SECTION("KT-FV")
  {
    using D = FV::Discretization<Probe, RectangularMeshSerial<1>>;
    check<D, FV::KurganovTadmor::Assembler<D>>(0, 3e-2);
  }
}
