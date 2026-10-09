#pragma once

// The KT headers come before DiFfRG.hh, see KT.cc.
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>
#include <DiFfRG/model/fv_boundaries.hh>

#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "flows_batched/flows.hh"

#include <vector>

/**
 * The O(N) model of model.hh with a batched flux: the flux at all points of a batch (the quadrature points of the
 * CG or DG assembler) is one map_points() call of the V integrator, on the backend chosen at compile time. Physics
 * and parameters are those of model.hh.
 */
namespace ON_batched
{
  struct Parameters {
    Parameters(const ConfigTree &config)
        : Lambda(config.get_double("/physical/Lambda")), N(config.get_double("/physical/N")), T(config.get_double("/physical/T")), lambda2(config.get_double("/physical/lambda2")),
          lambda4(config.get_double("/physical/lambda4")), lambda6(config.get_double("/physical/lambda6"))
    {
    }
    double Lambda, N, T, lambda2, lambda4, lambda6;
  };

  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"m2">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  constexpr auto idxf = FEFunctionDesc{};

  /// Where the V integral runs: TBB on the CPU, or the GPU in double or single precision.
  enum class Backend { TBB, GPU, GPU_float };

  /**
   * @tparam backend integrator of the batched flux; the per-point flux always uses the TBB one, as a
   * GPU get() must not be called from several threads at once.
   * @tparam batched whether the model evaluates its flux with map_points. Without it, evaluate_batch is
   * AbstractModel's default: the per-point flux in a flat parallel loop.
   */
  template <Backend backend = Backend::TBB, bool batched = true>
  class Model : public def::AbstractModel<Model<backend, batched>, Components>,
                public def::fRG,
                public def::LLFFlux<Model<backend, batched>>,
                public def::FlowBoundaries<Model<backend, batched>>,
                public def::AD<Model<backend, batched>>
  {
  public:
    static constexpr uint dim = 1;
    static constexpr bool batch_reads_hessians = false;

  protected:
    const Parameters prm;
    mutable ONFiniteTBatchedFlows flow_equations;

  public:
    Model(const ConfigTree &config) : def::fRG(config.get_double("/physical/Lambda")), prm(config), flow_equations(config)
    {
      flow_equations.set_k(Lambda);
      flow_equations.set_T(prm.T);
    }

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      const auto &rho = pos[0];
      values[idxf("m2")] = prm.lambda2 + prm.lambda4 * rho + prm.lambda6 * powr<2>(rho);
    }

    void set_time(double t_)
    {
      t = t_;
      k = std::exp(-t) * prm.Lambda;
      flow_equations.set_k(k);
    }

    template <typename NT, typename Solution> void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux, const Point<dim> &x, const Solution &sol) const
    {
      const auto rho = x[0];
      const auto &fe_functions = get<"fe_functions">(sol);
      const auto &fe_derivatives = get<"fe_derivatives">(sol);

      const auto m2Pi = fe_functions[idxf("m2")];
      const auto m2Sigma = fe_functions[idxf("m2")] + 2. * rho * fe_derivatives[idxf("m2")][0];

      flow_equations.V.get(flux[idxf("m2")][0], k, prm.N, prm.T, m2Pi, m2Sigma);
    }

    template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
    {
      if constexpr (!batched) {
        def::AbstractModel<Model, Components>::evaluate_batch(out, batch);
        return;
      }
      // The flux is the only term; the source is zero and never computed.
      if (!out.requested(Term::flux)) return;
      using NT = typename Batch::number_type;
      const size_t n = batch.size();
      const auto m2Pi = batch.values(idxf("m2"));
      const auto dm2 = batch.derivatives(idxf("m2"), 0);
      const auto rho = batch.coordinates(0);

      std::vector<NT> m2Sigma(n);
      for (size_t i = 0; i < n; ++i)
        m2Sigma[i] = m2Pi[i] + 2. * rho[i] * dm2[i];

      integrator().map_points(out.flux(idxf("m2"), 0), k, prm.N, prm.T, m2Pi, m2Sigma);
    }

    template <int dim, typename NumberType, typename Solution> void cell_indicator(NumberType &indicator, const Point<dim> & /*p*/, const Solution &sol) const
    {
      indicator = get<"fe_hessians">(sol)[idxf("m2")][0][0];
    }

    template <int dim, typename DataOut, typename Solutions> void readouts(DataOut &output, const Point<dim> &x, const Solutions &sol) const
    {
      const auto &fe_functions = get<"fe_functions">(sol);
      const auto &fe_derivatives = get<"fe_derivatives">(sol);

      const double rho = x[0];
      const double m2Pi = fe_functions[idxf("m2")];
      const double m2Sigma = fe_functions[idxf("m2")] + 2. * rho * fe_derivatives[idxf("m2")][0];

      auto out_file = output.table("data.csv");
      out_file.set_Lambda(Lambda);
      out_file.value("sigma [GeV]", std::sqrt(2. * rho));
      out_file.value("m^2_{pi} [GeV^2]", m2Pi);
      out_file.value("m^2_{sigma} [GeV^2]", m2Sigma);
      out_file.value("m_{pi} [GeV]", m2Pi > 0. ? std::sqrt(m2Pi) : 0.);
      out_file.value("m_{sigma} [GeV]", m2Sigma > 0. ? std::sqrt(m2Sigma) : 0.);
    }

    /// The V integrator of the chosen backend, for direct access (e.g. its map_points policy).
    auto &integrator() const
    {
      if constexpr (backend == Backend::TBB)
        return flow_equations.V;
      else if constexpr (backend == Backend::GPU)
        return flow_equations.V_GPU;
      else
        return flow_equations.V_GPU_f;
    }
  };

  using LDGFunctionDesc = FEFunctionDescriptor<Scalar<"dm2">>;
  using LDGComponents = ComponentDescriptor<FEFunctionDesc, VariableDescriptor<>, ExtractorDescriptor<>, LDGFunctionDesc>;
  constexpr auto idxl = LDGFunctionDesc{};

  template <typename M> using LDGFluxes = def::LDGUpDownFluxes<M, def::UpDownFlux<def::FlowDirections<0>, def::UpDown<def::from_right>>>;

  /**
   * The model of model_LDG.hh with a batched flux: m2' is the LDG level 1, built from m2 by an upwind flux, and
   * the main flux at all quadrature points is one map_points() call. Same template parameters as Model.
   */
  template <Backend backend = Backend::TBB, bool batched = true>
  class ModelLDG : public def::AbstractModel<ModelLDG<backend, batched>, LDGComponents>,
                   public def::fRG,
                   public def::LLFFlux<ModelLDG<backend, batched>>,
                   public LDGFluxes<ModelLDG<backend, batched>>,
                   public def::FlowBoundaries<ModelLDG<backend, batched>>,
                   public def::AD<ModelLDG<backend, batched>>
  {
  public:
    static constexpr uint dim = 1;
    using Components = LDGComponents;

  protected:
    const Parameters prm;
    mutable ONFiniteTBatchedFlows flow_equations;

  public:
    ModelLDG(const ConfigTree &config) : def::fRG(config.get_double("/physical/Lambda")), prm(config), flow_equations(config)
    {
      flow_equations.set_k(Lambda);
      flow_equations.set_T(prm.T);
      this->components().add_dependency(1, 0, 0, 0);
      this->components().set_jacobian_constant(1, 0);
    }

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      const auto &rho = pos[0];
      values[idxf("m2")] = prm.lambda2 + prm.lambda4 * rho + prm.lambda6 * powr<2>(rho);
    }

    void set_time(double t_)
    {
      t = t_;
      k = std::exp(-t) * prm.Lambda;
      flow_equations.set_k(k);
    }

    template <typename NT, typename Solution> void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux, const Point<dim> &x, const Solution &sol) const
    {
      const auto rho = x[0];
      const auto &fe_functions = get<"fe_functions">(sol);
      const auto &derivatives = get<"LDG1">(sol);
      const auto m2Pi = fe_functions[idxf("m2")];
      const auto m2Sigma = fe_functions[idxf("m2")] + 2. * rho * derivatives[idxl("dm2")];
      flow_equations.V.get(flux[idxf("m2")][0], k, prm.N, prm.T, m2Pi, m2Sigma);
    }

    template <uint submodel, typename NT, typename Variables>
    void ldg_flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(submodel)> &flux, const Point<dim> & /*pos*/, const Variables &u) const
    {
      flux[idxl("dm2")][0] = u[idxf("m2")];
    }

    template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
    {
      if constexpr (!batched) {
        def::AbstractModel<ModelLDG, Components>::evaluate_batch(out, batch);
        return;
      }
      // The flux is the only term; the source is zero and never computed.
      if (!out.requested(Term::flux)) return;
      using NT = typename Batch::number_type;
      const size_t n = batch.size();
      const auto m2Pi = batch.values(idxf("m2"));
      const auto dm2 = batch.ldg_values(1, idxl("dm2"));
      const auto rho = batch.coordinates(0);

      std::vector<NT> m2Sigma(n);
      for (size_t i = 0; i < n; ++i)
        m2Sigma[i] = m2Pi[i] + 2. * rho[i] * dm2[i];

      integrator().map_points(out.flux(idxf("m2"), 0), k, prm.N, prm.T, m2Pi, m2Sigma);
    }

    template <int dim, typename DataOut, typename Solutions> void readouts(DataOut &output, const Point<dim> &x, const Solutions &sol) const
    {
      const auto &fe_functions = get<"fe_functions">(sol);
      const auto &derivatives = get<"LDG1">(sol);
      const double rho = x[0];
      const double m2Pi = fe_functions[idxf("m2")];
      const double m2Sigma = fe_functions[idxf("m2")] + 2. * rho * derivatives[idxl("dm2")];

      auto out_file = output.table("data.csv");
      out_file.set_Lambda(Lambda);
      out_file.value("sigma [GeV]", std::sqrt(2. * rho));
      out_file.value("m^2_{pi} [GeV^2]", m2Pi);
      out_file.value("m^2_{sigma} [GeV^2]", m2Sigma);
      out_file.value("m_{pi} [GeV]", m2Pi > 0. ? std::sqrt(m2Pi) : 0.);
      out_file.value("m_{sigma} [GeV]", m2Sigma > 0. ? std::sqrt(m2Sigma) : 0.);
    }

    auto &integrator() const
    {
      if constexpr (backend == Backend::TBB)
        return flow_equations.V;
      else if constexpr (backend == Backend::GPU)
        return flow_equations.V_GPU;
      else
        return flow_equations.V_GPU_f;
    }
  };

  struct KTParameters {
    KTParameters(const ConfigTree &config)
        : Lambda(config.get_double("/physical/Lambda")), N(config.get_double("/physical/N")), T(config.get_double("/physical/T")), m2(config.get_double("/physical/m2")),
          lambda(config.get_double("/physical/lambda"))
    {
    }
    double Lambda, N, T, m2, lambda;
  };

  /**
   * The model of model_KT.hh (parameter_KT.toml) with a batched flux: the advection flux (pion loop) and the
   * diffusion flux (sigma loop) at all face traces of the KT assembler are one map_points() call each. Same
   * template parameters as Model; the GPU backends run both integrals in double.
   */
  template <Backend backend = Backend::TBB, bool batched = true>
  class ModelKT : public def::AbstractModel<ModelKT<backend, batched>, Components>,
                  public def::fRG,
                  public def::RhoSymmetricLinearExtrapolationBoundaries<ModelKT<backend, batched>>,
                  public def::AD<ModelKT<backend, batched>>
  {
  public:
    static constexpr uint dim = 1;

  protected:
    const KTParameters prm;
    mutable ONFiniteTBatchedFlows flow_equations;

  public:
    ModelKT(const ConfigTree &config) : def::fRG(config.get_double("/physical/Lambda")), prm(config), flow_equations(config)
    {
      flow_equations.set_k(Lambda);
      flow_equations.set_T(prm.T);
    }

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const { values[idxf("m2")] = prm.m2 + prm.lambda / 2. * pos[0]; }

    void set_time(double t_)
    {
      t = t_;
      k = std::exp(-t) * prm.Lambda;
      flow_equations.set_k(k);
    }

    /// Advection flux: (N-1) times the pion loop, which depends on m^2 only.
    template <typename NT, typename Solution> void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &F, const Point<dim> & /*x*/, const Solution &sol) const
    {
      NT pion_loop;
      flow_equations.V_pion.get(pion_loop, k, prm.N, prm.T, get<0>(sol)[idxf("m2")]);
      F[idxf("m2")][0] = (prm.N - 1.) * pion_loop;
    }

    /// Diffusion flux: the sigma loop, m^2_sigma = m^2 + 2 rho dm^2/drho.
    template <typename NT, typename Solution>
    void diffusion_flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &F, const Point<dim> &x, const Solution &sol) const
    {
      const auto m2Sigma = get<0>(sol)[idxf("m2")] + 2. * x[0] * get<1>(sol)[idxf("m2")][0];
      NT sigma_loop;
      flow_equations.V_sigma.get(sigma_loop, k, prm.N, prm.T, m2Sigma);
      F[idxf("m2")][0] = sigma_loop;
    }

    template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
    {
      if constexpr (!batched) {
        def::AbstractModel<ModelKT, Components>::evaluate_batch(out, batch);
        return;
      }
      using NT = typename Batch::number_type;
      const size_t n = batch.size();
      const auto m2 = batch.values(idxf("m2"));
      if (out.requested(Term::flux)) {
        const auto F = out.flux(idxf("m2"), 0);
        pion_integrator().map_points(F, k, prm.N, prm.T, m2);
        for (auto &f : F)
          f *= prm.N - 1.;
      }
      if (out.requested(Term::diffusion_flux)) {
        const auto dm2 = batch.derivatives(idxf("m2"), 0);
        const auto rho = batch.coordinates(0);
        std::vector<NT> m2Sigma(n);
        for (size_t i = 0; i < n; ++i)
          m2Sigma[i] = m2[i] + 2. * rho[i] * dm2[i];
        sigma_integrator().map_points(out.diffusion_flux(idxf("m2"), 0), k, prm.N, prm.T, m2Sigma);
      }
    }

    template <int dim, typename DataOut, typename Solutions> void readouts(DataOut &output, const Point<dim> &x, const Solutions &sol) const
    {
      const double rho = x[0];
      const double m2Pi = get<"fe_functions">(sol)[idxf("m2")];
      const double m2Sigma = m2Pi + 2. * rho * get<"fe_derivatives">(sol)[idxf("m2")][0];
      auto out_file = output.table("data.csv");
      out_file.set_Lambda(Lambda);
      out_file.value("sigma [GeV]", std::sqrt(2. * rho));
      out_file.value("m^2_{pi} [GeV^2]", m2Pi);
      out_file.value("m^2_{sigma} [GeV^2]", m2Sigma);
      out_file.value("m_{pi} [GeV]", m2Pi > 0. ? std::sqrt(m2Pi) : 0.);
      out_file.value("m_{sigma} [GeV]", m2Sigma > 0. ? std::sqrt(m2Sigma) : 0.);
    }

    auto &pion_integrator() const
    {
      if constexpr (backend == Backend::TBB)
        return flow_equations.V_pion;
      else
        return flow_equations.V_pion_GPU;
    }
    auto &sigma_integrator() const
    {
      if constexpr (backend == Backend::TBB)
        return flow_equations.V_sigma;
      else
        return flow_equations.V_sigma_GPU;
    }
  };
} // namespace ON_batched
