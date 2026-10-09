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
#include <type_traits>

// A model may need its extractors evaluated somewhere other than the EoM -- at a shock, say, whose
// position is not expressible as a pointwise EoM field. It says so by defining extractor_point().
// These tests pin down the two halves of that contract:
//
//   * extract() runs at the point the model asked for, and sees the solution *there*;
//   * readouts() keeps running at its own EoM, unaffected;
//
// and that a model which does not define extractor_point() behaves exactly as it did before the
// hook existed.

namespace
{
  using namespace dealii;
  using namespace DiFfRG;

  // u(x) = 2 (x - 0.2), so the readout EoMs below sit at known points and the solution value at any
  // point identifies that point uniquely.
  template <typename Derived, typename Components_>
  class ProbeBase : public def::AbstractModel<Derived, Components_>, public def::Time
  {
  public:
    using Components = Components_;

    /// Where extractor_point() sends the extractors, as an offset from the EoM.
    static constexpr double offset = 0.25;

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
      extractors[0] = get<"fe_functions">(sol)[0];
      // def::AD instantiates this with an AD number to build the extractor jacobian; only the
      // plain double pass is an observation worth recording.
      if constexpr (std::is_same_v<NT, double>) {
        extract_point = x[0];
        extract_u = get<"fe_functions">(sol)[0];
      }
    }

    template <typename DataOut, typename Solution>
    void capture_readout(DataOut &, const Point<1> &x, const Solution &sol) const
    {
      readout_point = x[0];
      readout_u = get<"fe_functions">(sol)[0];
      readout_extractor = get<"extractors">(sol)[0];
    }

    template <typename Helper, typename DataOut> void readouts_multiple(Helper &helper, DataOut &) const
    {
      helper(
          "primary", [](const auto &, const auto &values) { return std::array<double, 1>{{values[0] - 0.6}}; },
          [&](auto &output, const auto &x, const auto &sol) { capture_readout(output, x, sol); });
    }

    static constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    mutable double extract_point = nan;
    mutable double extract_u = nan;
    mutable double readout_point = nan;
    mutable double readout_u = nan;
    mutable double readout_extractor = nan;
  };

  /// Reads its extractors `offset` to the right of the EoM, using the sample only to confirm it is
  /// well formed -- the detector logic belongs to the models that need one, not to this test.
  template <typename Derived, typename Components_> class MovedProbeBase : public ProbeBase<Derived, Components_>
  {
  public:
    template <int dim, typename NT>
    Point<dim> extractor_point(const Point<dim> &EoM, const SolutionSample<dim, NT> &sample) const
    {
      sample_size = sample.size();
      sample_sorted = true;
      sample_widths_positive = true;
      for (size_t i = 0; i < sample.size(); ++i) {
        if (!(sample[i].cell_width > 0.)) sample_widths_positive = false;
        if (i > 0 && !(sample[i].point[0] > sample[i - 1].point[0])) sample_sorted = false;
      }

      Point<dim> moved(EoM);
      moved[0] += this->offset;
      return moved;
    }

    mutable size_t sample_size = 0;
    mutable bool sample_sorted = false;
    mutable bool sample_widths_positive = false;
  };

  using ProbeComponents =
      ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<>, ExtractorDescriptor<Scalar<"e">>>;
  using LDGProbeComponents = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<>,
                                                 ExtractorDescriptor<Scalar<"e">>, FEFunctionDescriptor<Scalar<"du">>>;

  template <typename Derived>
  class StandardBases : public def::LLFFlux<Derived>,
                        public def::FlowBoundaries<Derived>,
                        public def::FVDefaultBoundaries<Derived>,
                        public def::AD<Derived>
  {
  };

  class MovedProbe : public MovedProbeBase<MovedProbe, ProbeComponents>, public StandardBases<MovedProbe>
  {
  };
  class PlainProbe : public ProbeBase<PlainProbe, ProbeComponents>, public StandardBases<PlainProbe>
  {
  };

  template <typename Derived>
  class LDGBases : public def::NoNumFlux<Derived>, public def::FlowBoundaries<Derived>, public def::AD<Derived>
  {
  public:
    template <uint dependent, int dim, typename NT, typename Solutions_s, typename Solutions_n>
    void ldg_numflux(std::array<Tensor<1, dim, NT>, 1> &, const Tensor<1, dim> &, const Point<dim> &,
                     const Solutions_s &, const Solutions_n &) const
    {
    }
  };

  class MovedLDGProbe : public MovedProbeBase<MovedLDGProbe, LDGProbeComponents>, public LDGBases<MovedLDGProbe>
  {
  };
  class PlainLDGProbe : public ProbeBase<PlainLDGProbe, LDGProbeComponents>, public LDGBases<PlainLDGProbe>
  {
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

  template <typename Discretization, typename Assembler, typename Model>
  void run_one_frame(Model &model, const int fe_order)
  {
    auto json = make_json(fe_order);
    RectangularMeshSerial<1> mesh{Config::ConfigurationMesh<1>(json)};
    Discretization discretization(mesh, json);
    Assembler assembler(discretization, model, json);
    FE::FlowingVariables state(discretization);
    state.interpolate(model);

    auto output_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "extractor_point", "output");
    OutputSession<Discretization> output(output_path, json);
    output.write_frame(0., [&](auto &frame) {
      assembler.attach_data_output(frame, state.spatial_data(), typename Discretization::VectorType());
    });
  }

  /// u = 2 (x - 0.2) is exact for every discretization here, so the value at a point pins the point.
  double u_at(const double x) { return 2. * (x - 0.2); }

  template <typename Discretization, typename Assembler, typename Model>
  void check_moved(Model &model, const int fe_order, const double tol)
  {
    run_one_frame<Discretization, Assembler>(model, fe_order);

    // The readout is at the EoM, u = 0.6 -> x = 0.5, and is not moved by the hook.
    CHECK(model.readout_point == Catch::Approx(0.5).margin(tol));
    CHECK(model.readout_u == Catch::Approx(0.6).margin(tol));

    // The extractors are one `offset` to the right of it, and saw the solution there.
    CHECK(model.extract_point == Catch::Approx(0.5 + Model::offset).margin(tol));
    CHECK(model.extract_u == Catch::Approx(u_at(0.5 + Model::offset)).margin(tol));

    // ... and that is the value the readout was handed, not the one at its own point.
    CHECK(model.readout_extractor == Catch::Approx(model.extract_u).margin(1e-12));
    CHECK(model.readout_extractor != Catch::Approx(model.readout_u).margin(1e-3));

    // The sample the model was handed covers the mesh, is ordered, and has usable widths.
    CHECK(model.sample_size == 50);
    CHECK(model.sample_sorted);
    CHECK(model.sample_widths_positive);
  }

  template <typename Discretization, typename Assembler, typename Model>
  void check_plain(Model &model, const int fe_order, const double tol)
  {
    run_one_frame<Discretization, Assembler>(model, fe_order);

    // Without the hook, extractors are evaluated at the readout's own EoM -- the behaviour that
    // predates it.
    CHECK(model.readout_point == Catch::Approx(0.5).margin(tol));
    CHECK(model.extract_point == Catch::Approx(model.readout_point).margin(1e-12));
    CHECK(model.readout_extractor == Catch::Approx(model.readout_u).margin(1e-12));
  }
} // namespace

TEST_CASE("A model can move its extractors off the EoM", "[discretization][extractor-point]")
{
  try {
    spdlog::stdout_color_mt("log");
  } catch (const spdlog::spdlog_ex &) {
  }

  // FV is piecewise constant, so its extraction point is only resolved to a cell width.
  constexpr double fem_tol = 5e-3;
  constexpr double fv_tol = 3e-2;

  SECTION("CG")
  {
    MovedProbe model;
    using D = CG::Discretization<MovedProbe, RectangularMeshSerial<1>>;
    check_moved<D, CG::Assembler<D>>(model, 2, fem_tol);
  }
  SECTION("DG")
  {
    MovedProbe model;
    using D = DG::Discretization<MovedProbe, RectangularMeshSerial<1>>;
    check_moved<D, DG::Assembler<D>>(model, 1, fem_tol);
  }
  SECTION("LDG")
  {
    MovedLDGProbe model;
    using D = LDG::Discretization<MovedLDGProbe, RectangularMeshSerial<1>>;
    check_moved<D, LDG::Assembler<D>>(model, 1, fem_tol);
  }
  SECTION("KT-FV")
  {
    MovedProbe model;
    using D = FV::Discretization<MovedProbe, RectangularMeshSerial<1>>;
    check_moved<D, FV::KurganovTadmor::Assembler<D>>(model, 0, fv_tol);
  }
}

TEST_CASE("A model without extractor_point is unaffected", "[discretization][extractor-point]")
{
  try {
    spdlog::stdout_color_mt("log");
  } catch (const spdlog::spdlog_ex &) {
  }

  SECTION("CG")
  {
    PlainProbe model;
    using D = CG::Discretization<PlainProbe, RectangularMeshSerial<1>>;
    check_plain<D, CG::Assembler<D>>(model, 2, 5e-3);
  }
  SECTION("LDG")
  {
    PlainLDGProbe model;
    using D = LDG::Discretization<PlainLDGProbe, RectangularMeshSerial<1>>;
    check_plain<D, LDG::Assembler<D>>(model, 1, 5e-3);
  }
  SECTION("KT-FV")
  {
    PlainProbe model;
    using D = FV::Discretization<PlainProbe, RectangularMeshSerial<1>>;
    check_plain<D, FV::KurganovTadmor::Assembler<D>>(model, 0, 3e-2);
  }
}
