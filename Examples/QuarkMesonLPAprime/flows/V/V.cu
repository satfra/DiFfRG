#define FLOW_CODE

#include "V.hh"
#include "V.kernel"

namespace DiFfRG
{
  namespace Flows
  {
    V_integrator::V_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 1> grid_sizes,
                               const double x_extent, const JSONValue &json)
        : quadrature_provider(quadrature_provider), grid_sizes(grid_sizes), x_extent(x_extent),
          jacobian_quadrature_factor(json.get_double("/integration/jacobian_quadrature_factor")), json(json)
    {
      integrator = std::make_unique<DiFfRG::IntegratorTBB<3, double, V_kernel<__REGULATOR__>>>(
          quadrature_provider, grid_sizes, x_extent, json);
      for (uint i = 0; i < 1; ++i)
        jac_grid_sizes[i] = uint(jacobian_quadrature_factor * grid_sizes[i]);
      integrator_AD = std::make_unique<DiFfRG::IntegratorTBB<3, autodiff::real, V_kernel<__REGULATOR__>>>(
          quadrature_provider, jac_grid_sizes, x_extent, json);
    }

    V_integrator::V_integrator(const V_integrator &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes),
          jac_grid_sizes(other.jac_grid_sizes), x_extent(other.x_extent),
          jacobian_quadrature_factor(other.jacobian_quadrature_factor), json(other.json),
          integrator(std::make_unique<DiFfRG::IntegratorTBB<3, double, V_kernel<__REGULATOR__>>>(
              other.quadrature_provider, other.grid_sizes, other.x_extent, other.json)),
          integrator_AD(std::make_unique<DiFfRG::IntegratorTBB<3, autodiff::real, V_kernel<__REGULATOR__>>>(
              other.quadrature_provider, other.jac_grid_sizes, other.x_extent, other.json))
    {
    }

    V_integrator::~V_integrator() = default;

    std::future<double> V_integrator::request_CT(const double k, const double p0f, const double p, const double Nc,
                                                 const double Nf, const double T, const double muq, const double etaQ,
                                                 const double etaPhi, const double hPhi, const double d1V,
                                                 const double d2V, const double d3V, const double rhoPhi)
    {
      return integrator->request(k, p0f, p, Nc, Nf, T, muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
    }

    double V_integrator::get_CT(const double k, const double p0f, const double p, const double Nc, const double Nf,
                                const double T, const double muq, const double etaQ, const double etaPhi,
                                const double hPhi, const double d1V, const double d2V, const double d3V,
                                const double rhoPhi)
    {
      return integrator->get(k, p0f, p, Nc, Nf, T, muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
    }

    std::future<autodiff::real> V_integrator::request_AD(const double k, const double p0f, const double p,
                                                         const double Nc, const double Nf, const double T,
                                                         const double muq, const autodiff::real etaQ,
                                                         const autodiff::real etaPhi, const autodiff::real hPhi,
                                                         const autodiff::real d1V, const autodiff::real d2V,
                                                         const autodiff::real d3V, const double rhoPhi)
    {
      return integrator_AD->request(k, p0f, p, Nc, Nf, T, muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
    }

    autodiff::real V_integrator::get_AD(const double k, const double p0f, const double p, const double Nc,
                                        const double Nf, const double T, const double muq, const autodiff::real etaQ,
                                        const autodiff::real etaPhi, const autodiff::real hPhi,
                                        const autodiff::real d1V, const autodiff::real d2V, const autodiff::real d3V,
                                        const double rhoPhi)
    {
      return integrator_AD->get(k, p0f, p, Nc, Nf, T, muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
    }
  } // namespace Flows
} // namespace DiFfRG