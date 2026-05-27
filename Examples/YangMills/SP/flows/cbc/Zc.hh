#pragma once

template <typename REG> class Zc_kernel;
#include "../def.hh"

#include <future>
#include <memory>

namespace DiFfRG
{
  namespace Flows
  {
    class Zc_integrator
    {
    public:
      Zc_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 2> grid_sizes, const double x_extent,
                    const JSONValue &json);
      Zc_integrator(const Zc_integrator &other);
      ~Zc_integrator();

      template <typename NT, typename... T> std::future<NT> request(T &&...t)
      {
        static_assert(std::is_same_v<NT, double>, "Unknown type requested of Zc_integrator::request");
        if constexpr (std::is_same_v<NT, double>) return request_CT(std::forward<T>(t)...);
      }

      template <typename NT, typename... T> NT get(T &&...t)
      {
        static_assert(std::is_same_v<NT, double>, "Unknown type requested of Zc_integrator::request");
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
                                     const double m2A);
      double get_CT(const double k, const double p,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA3,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZAcbc,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA4,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZc,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &Zc,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZA,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA, const double m2A);

      QuadratureProvider &quadrature_provider;
      const std::array<uint, 2> grid_sizes;
      std::array<uint, 2> jac_grid_sizes;
      const double x_extent;
      const double jacobian_quadrature_factor;
      const JSONValue json;

      std::unique_ptr<DiFfRG::IntegratorAngleGPU<4, double, Zc_kernel<__REGULATOR__>>> integrator;
    };
  } // namespace Flows
} // namespace DiFfRG