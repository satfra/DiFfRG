#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/timestepping/ida_driver.hh>
#include <DiFfRG/timestepping/sundials_ida.hh>
#include <boilerplate/models.hh>

#include <cmath>
#include <string>

using namespace DiFfRG;

// After a restart IDA continues warm, at its order and step size (IDADriver). A restart that changes nothing must
// then leave the result as without restarts, and cost far less than a cold restart.

namespace
{
  constexpr uint dim = 1;

  /// du/dt = u
  using Model = Testing::ModelExp<dim>;

  using Discretization = CG::Discretization<Model, RectangularMeshSerial<dim>>;
  using Assembler = CG::Assembler<Discretization>;
  using VectorType = typename Discretization::VectorType;

  /// Asks IDA to restart once at every output, without changing the mesh.
  class RestartEveryOutput : public AbstractAdaptor<VectorType>
  {
  public:
    bool operator()(const double t, VectorType &) override { return restart(t); }
    bool adapt_at(const double t, VectorType &, const std::vector<VectorType *> &) override { return restart(t); }
    bool adapt(VectorType &) override { return true; }
    double last_adaptation_time() const override { return last; }
    void set_last_adaptation_time(const double t) override { last = t; }

  private:
    /// Once per time: the stepper asks again until the answer is no.
    bool restart(const double t)
    {
      if (t == last) return false;
      last = t;
      return true;
    }
    double last = -1.;
  };

  ConfigTree make_config(const bool warm_restart)
  {
    return json::value({{"physical", {{"Lambda", 1.}}},
                        {"discretization",
                         {{"fe_order", 1},
                          {"overintegration", 0},
                          {"EoM_abs_tol", 1e-12},
                          {"EoM_max_iter", 0},
                          {"grid", {{"x_grid", "0:0.1:1"}, {"refine", 0}}}}},
                        {"timestepping",
                         {{"final_time", 1.},
                          {"output_dt", 0.05},
                          {"implicit",
                           {{"dt", 1e-3},
                            {"minimal_dt", 1e-12},
                            {"maximal_dt", 0.1},
                            {"abs_tol", 1e-12},
                            {"rel_tol", 1e-9},
                            {"warm_restart", warm_restart}}}}},
                        {"output", {{"verbosity", 0}, {"vtk", false}, {"hdf5", false}}}});
  }

  struct Result {
    VectorType u;
    uint jacobians;
  };

  template <typename Adaptor> Result run(const bool warm_restart, Adaptor &&adaptor)
  {
    const auto config = make_config(warm_restart);
    const auto path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "ida_warm", "ida_warm");
    Testing::PhysicalParameters prm{};
    prm.initial_x0 = {{1., 0., 0.}};
    prm.initial_x1 = {{0.5, 0., 0.}};
    Model model(prm);
    RectangularMeshSerial<dim> mesh{Config::ConfigurationMesh<dim>(config)};
    Discretization discretization(mesh, config);
    Assembler assembler(discretization, model, config);
    OutputSession<Assembler> data_out(path, config);
    TimeStepperSUNDIALS_IDA<Assembler> stepper(config, assembler, data_out, adaptor);
    FE::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    stepper.run(state, 0., 1.);
    return {state.spatial_data(), assembler.num_jacobians()};
  }
} // namespace

TEST_CASE("IDA restarts warm without changing the result", "[timestepping][sundials_ida]")
{
  DiFfRG::Init();
  const auto plain = run(true, NoAdaptivity<VectorType>());
  const auto warm = run(true, RestartEveryOutput());
  const auto cold = run(false, RestartEveryOutput());

  // u(x, 1) = e (1 + x / 2)
  for (const auto *r : {&plain, &warm, &cold})
    for (std::size_t i = 0; i < r->u.size(); ++i)
      REQUIRE(std::isfinite(r->u[i]));
  auto deviation = warm.u;
  deviation -= plain.u;
  CHECK(deviation.linfty_norm() <= 1e-7 * plain.u.linfty_norm());

  if (IDADriver<VectorType>::warm_restart_available) {
    CAPTURE(plain.jacobians, warm.jacobians, cold.jacobians);
    CHECK(warm.jacobians < cold.jacobians / 2);
  }
}
