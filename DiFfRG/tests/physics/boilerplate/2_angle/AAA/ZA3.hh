#pragma once

#include "../def.hh"
#include "ZA3.kernel"

#include <future>
#include <memory>

namespace DiFfRG
{
  namespace Flows
  {
    template <template <typename, typename> class INT> class ZA3_integrator
    {
    public:
      ZA3_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 3> grid_sizes, const double x_extent,
                     const JSONValue &json)
          : quadrature_provider(quadrature_provider), grid_sizes(grid_sizes), x_extent(x_extent),
            jacobian_quadrature_factor(json.get_double("/integration/jacobian_quadrature_factor")), json(json)
      {
        integrator =
            std::make_unique<INT<double, ZA3_kernel<__REGULATOR__>>>(quadrature_provider, grid_sizes, x_extent, json);
      }

      ~ZA3_integrator() = default;

      template <typename NT, typename... T> std::future<NT> request(T &&...t)
      {
        static_assert(std::is_same_v<NT, double>, "Unknown type requested of ZA3_integrator::request");
        if constexpr (std::is_same_v<NT, double>) return request_CT(std::forward<T>(t)...);
      }

      template <typename NT, typename... T> NT get(T &&...t)
      {
        static_assert(std::is_same_v<NT, double>, "Unknown type requested of ZA3_integrator::request");
        if constexpr (std::is_same_v<NT, double>) return get_CT(std::forward<T>(t)...);
      }

    private:
      std::future<double> request_CT(const double k, const double p,
                                     const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA3,
                                     const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZAcbc,
                                     const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA4,
                                     const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZc,
                                     const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &Zc,
                                     const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZA,
                                     const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA,
                                     const double m2A)
      {
        return integrator->request(k, p, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA, m2A);
      }

      double get_CT(const double k, const double p,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA3,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZAcbc,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA4,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZc,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &Zc,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZA,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA, const double m2A)
      {
        return integrator->get(k, p, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA, m2A);
      }

      QuadratureProvider &quadrature_provider;
      const std::array<uint, 3> grid_sizes;
      std::array<uint, 3> jac_grid_sizes;
      const double x_extent;
      const double jacobian_quadrature_factor;
      const JSONValue json;

      std::unique_ptr<INT<double, ZA3_kernel<__REGULATOR__>>> integrator;
    };

  } // namespace Flows
} // namespace DiFfRG