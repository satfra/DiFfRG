#define FLOW_CODE

#include "hPhi0.hh"
#include "hPhi0.kernel"

namespace DiFfRG
{
  namespace Flows
  {
    hPhi0_integrator::hPhi0_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 2> grid_sizes,
                                       const double x_extent, const JSONValue &json)
        : quadrature_provider(quadrature_provider), grid_sizes(grid_sizes), x_extent(x_extent),
          jacobian_quadrature_factor(json.get_double("/integration/jacobian_quadrature_factor")), json(json)
    {
      integrator = std::make_unique<DiFfRG::IntegratorAngleFiniteTq0TBB<4, double, hPhi0_kernel<__REGULATOR__>>>(
          quadrature_provider, grid_sizes, x_extent, json);
    }

    hPhi0_integrator::hPhi0_integrator(const hPhi0_integrator &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes),
          jac_grid_sizes(other.jac_grid_sizes), x_extent(other.x_extent),
          jacobian_quadrature_factor(other.jacobian_quadrature_factor), json(other.json),
          integrator(std::make_unique<DiFfRG::IntegratorAngleFiniteTq0TBB<4, double, hPhi0_kernel<__REGULATOR__>>>(
              other.quadrature_provider, other.grid_sizes, other.x_extent, other.json))
    {
    }

    hPhi0_integrator::~hPhi0_integrator() = default;

    void hPhi0_integrator::set_T(const double T, const double E) { integrator->set_T(T, E); }

    std::future<double> hPhi0_integrator::request_CT(const double k, const double p0f, const double p, const double Nc,
                                                     const double Nf, const double T, const double muq,
                                                     const double etaQ, const double etaPhi, const double hPhi,
                                                     const double d1V, const double d2V, const double d3V,
                                                     const double rhoPhi)
    {
      return integrator->request(k, p0f, p, Nc, Nf, T, muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
    }

    double hPhi0_integrator::get_CT(const double k, const double p0f, const double p, const double Nc, const double Nf,
                                    const double T, const double muq, const double etaQ, const double etaPhi,
                                    const double hPhi, const double d1V, const double d2V, const double d3V,
                                    const double rhoPhi)
    {
      return integrator->get(k, p0f, p, Nc, Nf, T, muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
    }

  } // namespace Flows
} // namespace DiFfRG