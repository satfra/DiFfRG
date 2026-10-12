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

struct Parameters {
  Parameters(const ConfigTree &config)
      : Lambda(config.get_double("/physical/Lambda")), N(config.get_double("/physical/N")),
        m2(config.get_double("/physical/m2")), lambda(config.get_double("/physical/lambda"))
  {
  }
  double Lambda, N, m2, lambda;
};

using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
using Components = ComponentDescriptor<FEFunctionDesc>;
constexpr auto idxf = FEFunctionDesc{};

/**
 * @brief What all discretizations share: the O(N) model in LPA, d = 3, Litim regulator.
 *
 * The FE function is u = dV/drho. With R_k(q) = (k^2 - q^2) theta(k^2 - q^2), a mode of mass m^2 contributes
 * loop(m^2) = k^5 / (6 pi^2 (k^2 + m^2)) to the flow of V, so that
 *
 *   d_t u = d_rho [ (N - 1) loop(m^2_pi) + loop(m^2_sigma) ],   m^2_pi = u,   m^2_sigma = u + 2 rho d_rho u.
 */
class ONCommon : public def::fRG
{
public:
  static constexpr uint dim = 1;

  ONCommon(const ConfigTree &config) : def::fRG(config.get_double("/physical/Lambda")), prm(config) {}

  template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
  {
    values[idxf("u")] = prm.m2 + prm.lambda / 2. * pos[0];
  }

  /// The condensate and the curvature masses at the minimum of the potential.
  template <int dim_, typename DataOut, typename Solutions>
  void readouts(DataOut &output, const Point<dim_> &x, const Solutions &sol) const
  {
    const double rho = x[0];
    const double m2Pi = get<"fe_functions">(sol)[idxf("u")];
    const double m2Sigma = m2Pi + 2. * rho * get<"fe_derivatives">(sol)[idxf("u")][0];
    auto table = output.table("data.csv");
    table.value("sigma", std::sqrt(2. * rho));
    table.value("m2Pi", m2Pi);
    table.value("m2Sigma", m2Sigma);
  }

protected:
  const Parameters prm;

  template <typename NT> NT loop(const NT m2) const { return powr<5>(k) / (6. * powr<2>(M_PI) * (k2 + m2)); }
};

/**
 * @brief The model for CG and DG: one flux carrying both modes. DG additionally needs the numerical flux between
 * cells, here local Lax-Friedrichs; CG ignores it.
 */
class ON_FEM : public def::AbstractModel<ON_FEM, Components>,
               public ONCommon,
               public def::LLFFlux<ON_FEM>,
               public def::FlowBoundaries<ON_FEM>,
               public def::AD<ON_FEM>
{
public:
  ON_FEM(const ConfigTree &config) : ONCommon(config) {}

  // AbstractModel has defaults for these, too; the shared ones win.
  using ONCommon::initial_condition;
  using ONCommon::readouts;

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, dim, NT>, 1> &F, const Point<dim> &x, const Solution &sol) const
  {
    const auto m2Pi = get<"fe_functions">(sol)[idxf("u")];
    const auto m2Sigma = m2Pi + 2. * x[0] * get<"fe_derivatives">(sol)[idxf("u")][0];
    F[idxf("u")][0] = (prm.N - 1.) * loop(m2Pi) + loop(m2Sigma);
  }
};

/**
 * @brief The model for the Kurganov-Tadmor finite-volume scheme. The flux is split: the Goldstone modes depend on u
 * alone and form the advective flux, the radial mode reads d_rho u and forms the diffusive flux. Boundaries are
 * ghost cells, here even in rho at the origin and linearly extrapolated at the far end.
 */
class ON_KT : public def::AbstractModel<ON_KT, Components>,
              public ONCommon,
              public def::RhoSymmetricLinearExtrapolationBoundaries<ON_KT>,
              public def::AD<ON_KT>
{
public:
  ON_KT(const ConfigTree &config) : ONCommon(config) {}

  using ONCommon::initial_condition;
  using ONCommon::readouts;

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, dim, NT>, 1> &F, const Point<dim> & /*x*/, const Solution &sol) const
  {
    F[idxf("u")][0] = (prm.N - 1.) * loop(get<"fe_functions">(sol)[idxf("u")]);
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, dim, NT>, 1> &F, const Point<dim> &x, const Solution &sol) const
  {
    const auto m2Sigma = get<"fe_functions">(sol)[idxf("u")] + 2. * x[0] * get<"fe_derivatives">(sol)[idxf("u")][0];
    F[idxf("u")][0] = loop(m2Sigma);
  }
};
