#pragma once

template <typename REG> class ZA4SP_kernel;
#include "../def.hh"

#include <future>
#include <memory>

namespace DiFfRG
{
  namespace Flows
  {
    class ZA4SP_integrator
    {
    public:
      ZA4SP_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 4> grid_sizes, const double x_extent, const ConfigTree &json);
      ZA4SP_integrator(const ZA4SP_integrator &other);
      ~ZA4SP_integrator();

      template <typename NT, typename... T> std::future<NT> request(T &&...t)
      {
        static_assert(std::is_same_v<NT, double>, "Unknown type requested of ZA4SP_integrator::request");
        if constexpr (std::is_same_v<NT, double>) return request_CT(std::forward<T>(t)...);
      }

      template <typename NT, typename... T> NT get(T &&...t)
      {
        static_assert(std::is_same_v<NT, double>, "Unknown type requested of ZA4SP_integrator::request");
        if constexpr (std::is_same_v<NT, double>) return get_CT(std::forward<T>(t)...);
      }

    private:
      std::future<double>
      request_CT(const double k, const double p,
                 const TexLinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>> &ZA3,
                 const TexLinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>> &ZAcbc,
                 const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA4SP,
                 const TexLinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>> &ZA4tadpole,
                 const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZc, const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &Zc,
                 const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZA, const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA,
                 const double m2A);
      double get_CT(const double k, const double p,
                    const TexLinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>> &ZA3,
                    const TexLinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>> &ZAcbc,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA4SP,
                    const TexLinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>> &ZA4tadpole,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZc, const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &Zc,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZA, const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA,
                    const double m2A);

      QuadratureProvider &quadrature_provider;
      const std::array<uint, 4> grid_sizes;
      std::array<uint, 4> jac_grid_sizes;
      const double x_extent;
      const double jacobian_quadrature_factor;
      const ConfigTree json;

      std::unique_ptr<DiFfRG::Integrator4DGPU<double, ZA4SP_kernel<__REGULATOR__>>> integrator;
    };
  } // namespace Flows
} // namespace DiFfRG