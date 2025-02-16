#pragma once

template <typename REG> class hPhi0_kernel;
#include "../def.hh"

#include <future>
#include <memory>

namespace DiFfRG
{
  namespace Flows
  {
    class hPhi0_integrator
    {
    public:
      hPhi0_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 2> grid_sizes, const double x_extent,
                       const JSONValue &json);
      hPhi0_integrator(const hPhi0_integrator &other);
      ~hPhi0_integrator();

      template <typename NT, typename... T> std::future<NT> request(T &&...t)
      {
        static_assert(std::is_same_v<NT, double>, "Unknown type requested of hPhi0_integrator::request");
        if constexpr (std::is_same_v<NT, double>) return request_CT(std::forward<T>(t)...);
      }

      template <typename NT, typename... T> NT get(T &&...t)
      {
        static_assert(std::is_same_v<NT, double>, "Unknown type requested of hPhi0_integrator::request");
        if constexpr (std::is_same_v<NT, double>) return get_CT(std::forward<T>(t)...);
      }

      void set_T(const double T, const double E = 0);

    private:
      std::future<double> request_CT(const double k, const double p0f, const double p, const double Nc, const double Nf,
                                     const double T, const double muq, const double etaQ, const double etaPhi,
                                     const double hPhi, const double d1V, const double d2V, const double d3V,
                                     const double rhoPhi);
      double get_CT(const double k, const double p0f, const double p, const double Nc, const double Nf, const double T,
                    const double muq, const double etaQ, const double etaPhi, const double hPhi, const double d1V,
                    const double d2V, const double d3V, const double rhoPhi);

      QuadratureProvider &quadrature_provider;
      const std::array<uint, 2> grid_sizes;
      std::array<uint, 2> jac_grid_sizes;
      const double x_extent;
      const double jacobian_quadrature_factor;
      const JSONValue json;

      std::unique_ptr<DiFfRG::IntegratorAngleFiniteTq0TBB<4, double, hPhi0_kernel<__REGULATOR__>>> integrator;
    };
  } // namespace Flows
} // namespace DiFfRG