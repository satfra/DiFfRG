#pragma once

// The FV headers come before DiFfRG.hh: inside KurganovTadmor.hh, an unqualified Quadrature<dim> must resolve to
// dealii::Quadrature, not to DiFfRG::Quadrature, which DiFfRG.hh brings in.
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>
#include <DiFfRG/model/fv_boundaries.hh>

#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "flows/flows.hh"

#include <vector>

struct Parameters {
  Parameters(const ConfigTree &config)
      : Lambda(config.get_double("/physical/Lambda")), N(config.get_double("/physical/N")),
        T(config.get_double("/physical/T")), m2(config.get_double("/physical/m2")),
        lambda(config.get_double("/physical/lambda"))
  {
  }
  double Lambda, N, T, m2, lambda;
};

using FEFunctionDesc = FEFunctionDescriptor<Scalar<"m2">>;
using Components = ComponentDescriptor<FEFunctionDesc>;
constexpr auto idxf = FEFunctionDesc{};

/// How the model evaluates its momentum integrals: per point (the default evaluate_batch), or batched with
/// map_points on the CPU or on the GPU.
enum class Backend { per_point, cpu, gpu };

/**
 * @brief What the CG and the KT model share: parameters, the RG scale, the flows and the initial condition
 * V'(rho) = m2 + lambda / 2 * rho.
 */
class ONCommon : public def::fRG
{
public:
  ONCommon(const ConfigTree &config)
      : def::fRG(config.get_double("/physical/Lambda")), prm(config), flow_equations(config)
  {
    flow_equations.set_k(Lambda);
    flow_equations.set_T(prm.T);
  }

  void set_time(const double t_)
  {
    t = t_;
    k = std::exp(-t) * prm.Lambda;
    flow_equations.set_k(k);
  }

  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[idxf("m2")] = prm.m2 + prm.lambda / 2. * pos[0];
  }

  template <int dim, typename DataOut, typename Solutions>
  void readouts(DataOut &output, const Point<dim> &x, const Solutions &sol) const
  {
    const double rho = x[0];
    const double m2Pi = get<"fe_functions">(sol)[idxf("m2")];
    const double m2Sigma = m2Pi + 2. * rho * get<"fe_derivatives">(sol)[idxf("m2")][0];
    auto out_file = output.table("data.csv");
    out_file.set_Lambda(Lambda);
    out_file.value("sigma [GeV]", std::sqrt(2. * rho));
    out_file.value("m^2_{pi} [GeV^2]", m2Pi);
    out_file.value("m^2_{sigma} [GeV^2]", m2Sigma);
  }

protected:
  const Parameters prm;
  // mutable: the integrators' get() and map_points() are not const, while the model's callbacks are.
  mutable ONFlows flow_equations;
};

/**
 * @brief The O(N) model for the CG assembler. The flux is the full loop V(m2Pi, m2Sigma), with
 * m2Sigma = m2Pi + 2 rho dm2Pi/drho; there is no source.
 */
template <Backend backend>
class ON_CG : public def::AbstractModel<ON_CG<backend>, Components>,
              public ONCommon,
              public def::LLFFlux<ON_CG<backend>>,
              public def::FlowBoundaries<ON_CG<backend>>,
              public def::AD<ON_CG<backend>>
{
public:
  static constexpr uint dim = 1;
  // The flux reads values and first derivatives, never hessians: the assembler then skips them, in the batch and
  // in the AD jacobian.
  static constexpr bool batch_reads_hessians = false;

  ON_CG(const ConfigTree &config) : ONCommon(config) {}

  // AbstractModel has defaults for these, too; the shared ones win.
  using ONCommon::initial_condition;
  using ONCommon::readouts;

  /// The flux at one point, as in Tutorial 3. With Backend::per_point the default evaluate_batch calls it for
  /// every point, from many threads at once, so it uses the CPU integrator.
  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, dim, NT>, 1> &F, const Point<dim> &x, const Solution &sol) const
  {
    const auto m2Pi = get<"fe_functions">(sol)[idxf("m2")];
    const auto m2Sigma = m2Pi + 2. * x[0] * get<"fe_derivatives">(sol)[idxf("m2")][0];
    flow_equations.V.get(F[idxf("m2")][0], k, prm.N, prm.T, m2Pi, m2Sigma);
  }

  /// The flux at all points of a batch: one map_points call.
  template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
  {
    if constexpr (backend == Backend::per_point)
      return def::AbstractModel<ON_CG, Components>::evaluate_batch(out, batch);
    else {
      // Cell points ask for flux and source, faces for the flux only; this model has no source.
      if (!out.requested(Term::flux)) return;
      // double, or an AD number when the assembler builds the jacobian.
      using NT = typename Batch::number_type;
      const auto m2Pi = batch.values(idxf("m2"));
      const auto dm2Pi = batch.derivatives(idxf("m2"), 0);
      const auto rho = batch.coordinates(0);
      std::vector<NT> m2Sigma(batch.size());
      for (size_t i = 0; i < batch.size(); ++i)
        m2Sigma[i] = m2Pi[i] + 2. * rho[i] * dm2Pi[i];
      // k, N and T are shared by all points, m2Pi and m2Sigma have one value per point.
      if constexpr (backend == Backend::cpu)
        flow_equations.V.map_points(out.flux(idxf("m2"), 0), k, prm.N, prm.T, m2Pi, m2Sigma);
      else
        flow_equations.V_GPU.map_points(out.flux(idxf("m2"), 0), k, prm.N, prm.T, m2Pi, m2Sigma);
    }
  }
};

/**
 * @brief The O(N) model for the Kurganov-Tadmor assembler. The pion loop, which depends on m2 alone, is the
 * advection flux; the sigma loop, which also reads dm2/drho, is the diffusion flux.
 */
template <Backend backend>
class ON_KT : public def::AbstractModel<ON_KT<backend>, Components>,
              public ONCommon,
              public def::RhoSymmetricLinearExtrapolationBoundaries<ON_KT<backend>>,
              public def::AD<ON_KT<backend>>
{
public:
  static constexpr uint dim = 1;

  ON_KT(const ConfigTree &config) : ONCommon(config) {}

  // AbstractModel has defaults for these, too; the shared ones win.
  using ONCommon::initial_condition;
  using ONCommon::readouts;

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, dim, NT>, 1> &F, const Point<dim> & /*x*/, const Solution &sol) const
  {
    NT pion_loop;
    flow_equations.V_pion.get(pion_loop, k, prm.N, prm.T, get<"fe_functions">(sol)[idxf("m2")]);
    F[idxf("m2")][0] = (prm.N - 1.) * pion_loop;
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, dim, NT>, 1> &F, const Point<dim> &x, const Solution &sol) const
  {
    const auto m2Sigma = get<"fe_functions">(sol)[idxf("m2")] + 2. * x[0] * get<"fe_derivatives">(sol)[idxf("m2")][0];
    flow_equations.V_sigma.get(F[idxf("m2")][0], k, prm.N, prm.T, m2Sigma);
  }

  /// KT calls this once per term: the face traces for the flux, again for the diffusion flux, and the cells for
  /// the source (zero here).
  template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
  {
    if constexpr (backend == Backend::per_point)
      return def::AbstractModel<ON_KT, Components>::evaluate_batch(out, batch);
    else {
      using NT = typename Batch::number_type;
      // The CPU and the GPU integrators are different types, hence if constexpr.
      auto &pion = [&]() -> auto & {
        if constexpr (backend == Backend::cpu)
          return flow_equations.V_pion;
        else
          return flow_equations.V_pion_GPU;
      }();
      auto &sigma = [&]() -> auto & {
        if constexpr (backend == Backend::cpu)
          return flow_equations.V_sigma;
        else
          return flow_equations.V_sigma_GPU;
      }();
      const auto m2 = batch.values(idxf("m2"));
      if (out.requested(Term::flux)) {
        const auto F = out.flux(idxf("m2"), 0);
        pion.map_points(F, k, prm.N, prm.T, m2);
        for (auto &f : F)
          f *= prm.N - 1.;
      }
      if (out.requested(Term::diffusion_flux)) {
        const auto dm2 = batch.derivatives(idxf("m2"), 0);
        const auto rho = batch.coordinates(0);
        std::vector<NT> m2Sigma(batch.size());
        for (size_t i = 0; i < batch.size(); ++i)
          m2Sigma[i] = m2[i] + 2. * rho[i] * dm2[i];
        sigma.map_points(out.diffusion_flux(idxf("m2"), 0), k, prm.N, prm.T, m2Sigma);
      }
    }
  }
};
