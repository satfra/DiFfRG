#define FLOW_CODE

#include "lambda6.hh"
#include "lambda6.kernel"

namespace DiFfRG
{
  namespace Flows
  {
    lambda6_integrator::lambda6_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 2> grid_sizes,
                                           const double x_extent, const double q0_extent, const uint q0_summands,
                                           const ConfigTree &json)
        : quadrature_provider(quadrature_provider), grid_sizes(grid_sizes), x_extent(x_extent), q0_extent(q0_extent),
          q0_summands(q0_summands),
          jacobian_quadrature_factor(json.get_double("/integration/jacobian_quadrature_factor")), json(json)
    {
      integrator = std::make_unique<DiFfRG::IntegratorFiniteTq0TBB<4, double, lambda6_kernel<__REGULATOR__>>>(
          quadrature_provider, grid_sizes, x_extent, q0_extent, q0_summands, json);
    }

    lambda6_integrator::lambda6_integrator(const lambda6_integrator &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes),
          jac_grid_sizes(other.jac_grid_sizes), x_extent(other.x_extent), q0_extent(other.q0_extent),
          q0_summands(other.q0_summands), jacobian_quadrature_factor(other.jacobian_quadrature_factor),
          json(other.json),
          integrator(std::make_unique<DiFfRG::IntegratorFiniteTq0TBB<4, double, lambda6_kernel<__REGULATOR__>>>(
              other.quadrature_provider, other.grid_sizes, other.x_extent, other.q0_extent, other.q0_summands,
              other.json))
    {
    }

    lambda6_integrator::~lambda6_integrator() = default;

    void lambda6_integrator::set_T(const double value) { integrator->set_T(value); }
    void lambda6_integrator::set_q0_extent(const double value) { integrator->set_q0_extent(value); }

    std::future<double> lambda6_integrator::request_CT(
        const double k, const double p0f, const double T, const double muq, const double gAqbq1, const double etaA,
        const double etaQ, const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &MQ,
        const double lambda1, const double lambda2, const double lambda3, const double lambda4, const double lambda5,
        const double lambda6, const double lambda7, const double lambda8, const double lambda9, const double lambda10)
    {
      return integrator->request(k, p0f, T, muq, gAqbq1, etaA, etaQ, MQ, lambda1, lambda2, lambda3, lambda4, lambda5,
                                 lambda6, lambda7, lambda8, lambda9, lambda10);
    }

    double lambda6_integrator::get_CT(const double k, const double p0f, const double T, const double muq,
                                      const double gAqbq1, const double etaA, const double etaQ,
                                      const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &MQ,
                                      const double lambda1, const double lambda2, const double lambda3,
                                      const double lambda4, const double lambda5, const double lambda6,
                                      const double lambda7, const double lambda8, const double lambda9,
                                      const double lambda10)
    {
      return integrator->get(k, p0f, T, muq, gAqbq1, etaA, etaQ, MQ, lambda1, lambda2, lambda3, lambda4, lambda5,
                             lambda6, lambda7, lambda8, lambda9, lambda10);
    }

  } // namespace Flows
} // namespace DiFfRG