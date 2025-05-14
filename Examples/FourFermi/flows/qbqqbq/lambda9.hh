#pragma once

template <typename REG> class lambda9_kernel;
#include "../def.hh"

#include <future>
#include <memory>

namespace DiFfRG
{
  namespace Flows
  {
    class lambda9_integrator
    {
    public:
      lambda9_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 1> grid_sizes, const double x_extent,
                         const JSONValue &json);
      lambda9_integrator(const lambda9_integrator &other);
      ~lambda9_integrator();

      template <typename NT, typename... T> std::future<NT> request(T &&...t)
      {
        static_assert(std::is_same_v<NT, double>, "Unknown type requested of lambda9_integrator::request");
        if constexpr (std::is_same_v<NT, double>) return request_CT(std::forward<T>(t)...);
      }

      template <typename NT, typename... T> NT get(T &&...t)
      {
        static_assert(std::is_same_v<NT, double>, "Unknown type requested of lambda9_integrator::request");
        if constexpr (std::is_same_v<NT, double>) return get_CT(std::forward<T>(t)...);
      }

      void set_T(const double T, const double E = 0);

    private:
      std::future<double> request_CT(const double k, const double p0f, const double T, const double muq,
                                     const double gAqbq1, const double etaA, const double etaQ,
                                     const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &MQ,
                                     const double lambda1, const double lambda2, const double lambda3,
                                     const double lambda4, const double lambda5, const double lambda6,
                                     const double lambda7, const double lambda8, const double lambda9,
                                     const double lambda10);
      double get_CT(const double k, const double p0f, const double T, const double muq, const double gAqbq1,
                    const double etaA, const double etaQ,
                    const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &MQ, const double lambda1,
                    const double lambda2, const double lambda3, const double lambda4, const double lambda5,
                    const double lambda6, const double lambda7, const double lambda8, const double lambda9,
                    const double lambda10);

      QuadratureProvider &quadrature_provider;
      const std::array<uint, 1> grid_sizes;
      std::array<uint, 1> jac_grid_sizes;
      const double x_extent;
      const double jacobian_quadrature_factor;
      const JSONValue json;

      std::unique_ptr<DiFfRG::IntegratorFiniteTq0TBB<4, double, lambda9_kernel<__REGULATOR__>>> integrator;
    };
  } // namespace Flows
} // namespace DiFfRG