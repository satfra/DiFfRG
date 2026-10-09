#define FLOW_CODE

#include "ZA3.hh"
#include "ZA3.kernel"

namespace DiFfRG
{
  namespace Flows
  {
    ZA3_integrator::ZA3_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 3> grid_sizes, const double x_extent, const ConfigTree &json)
        : quadrature_provider(quadrature_provider), grid_sizes(grid_sizes), x_extent(x_extent),
          jacobian_quadrature_factor(json.get_double("/integration/jacobian_quadrature_factor")), json(json)
    {
      integrator = std::make_unique<DiFfRG::Integrator4D2AngGPU<double, ZA3_kernel<__REGULATOR__>>>(quadrature_provider, grid_sizes, x_extent, json);
    }

    ZA3_integrator::ZA3_integrator(const ZA3_integrator &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes), jac_grid_sizes(other.jac_grid_sizes), x_extent(other.x_extent),
          jacobian_quadrature_factor(other.jacobian_quadrature_factor), json(other.json),
          integrator(std::make_unique<DiFfRG::Integrator4D2AngGPU<double, ZA3_kernel<__REGULATOR__>>>(other.quadrature_provider, other.grid_sizes, other.x_extent, other.json))
    {
    }

    ZA3_integrator::~ZA3_integrator() = default;

    std::future<double> ZA3_integrator::request_CT(
        const double k, const double S0, const double S1, const double SPhi,
        const TexLinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>> &ZA3,
        const TexLinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>> &ZAcbc,
        const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA4SP,
        const TexLinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>> &ZA4tadpole,
        const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZc, const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &Zc,
        const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZA, const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA, const double m2A)
    {
      return integrator->request(k, S0, S1, SPhi, ZA3, ZAcbc, ZA4SP, ZA4tadpole, dtZc, Zc, dtZA, ZA, m2A);
    }

    double ZA3_integrator::get_CT(
        const double k, const double S0, const double S1, const double SPhi,
        const TexLinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>> &ZA3,
        const TexLinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>> &ZAcbc,
        const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA4SP,
        const TexLinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>> &ZA4tadpole,
        const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZc, const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &Zc,
        const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZA, const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA, const double m2A)
    {
      return integrator->get(k, S0, S1, SPhi, ZA3, ZAcbc, ZA4SP, ZA4tadpole, dtZc, Zc, dtZA, ZA, m2A);
    }

  } // namespace Flows
} // namespace DiFfRG