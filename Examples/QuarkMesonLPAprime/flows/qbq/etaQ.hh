#pragma once

template <typename REG> class etaQ_kernel;
#include "../def.hh"

#include <future>
#include <memory>

namespace DiFfRG
{
  namespace Flows
  {
    class etaQ_integrator
    {
    public:
      etaQ_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 3> grid_sizes, const double x_extent, const double q0_extent, const uint q0_summands,
                      const JSONValue &json);
      etaQ_integrator(const etaQ_integrator &other);
      ~etaQ_integrator();

      template <typename NT, typename... T> std::future<NT> request(T &&...t)
      {
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, autodiff::real>, "Unknown type requested of etaQ_integrator::request");
        if constexpr (std::is_same_v<NT, double>)
          return request_CT(std::forward<T>(t)...);
        else if constexpr (std::is_same_v<NT, autodiff::real>)
          return request_AD(std::forward<T>(t)...);
      }

      template <typename NT, typename... T> NT get(T &&...t)
      {
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, autodiff::real>, "Unknown type requested of etaQ_integrator::request");
        if constexpr (std::is_same_v<NT, double>)
          return get_CT(std::forward<T>(t)...);
        else if constexpr (std::is_same_v<NT, autodiff::real>)
          return get_AD(std::forward<T>(t)...);
      }

      void set_T(const double value);
      void set_q0_extent(const double value);

    private:
      std::future<double> request_CT(const double k, const double p0f, const double p, const double Nc, const double Nf, const double T, const double muq, const double etaQ,
                                     const double etaPhi, const double hPhi, const double d1V, const double d2V, const double d3V, const double rhoPhi);
      double get_CT(const double k, const double p0f, const double p, const double Nc, const double Nf, const double T, const double muq, const double etaQ, const double etaPhi,
                    const double hPhi, const double d1V, const double d2V, const double d3V, const double rhoPhi);
      std::future<autodiff::real> request_AD(const double k, const double p0f, const double p, const double Nc, const double Nf, const double T, const double muq,
                                             const autodiff::real etaQ, const autodiff::real etaPhi, const autodiff::real hPhi, const autodiff::real d1V, const autodiff::real d2V,
                                             const autodiff::real d3V, const double rhoPhi);
      autodiff::real get_AD(const double k, const double p0f, const double p, const double Nc, const double Nf, const double T, const double muq, const autodiff::real etaQ,
                            const autodiff::real etaPhi, const autodiff::real hPhi, const autodiff::real d1V, const autodiff::real d2V, const autodiff::real d3V,
                            const double rhoPhi);

      QuadratureProvider &quadrature_provider;
      const std::array<uint, 3> grid_sizes;
      std::array<uint, 3> jac_grid_sizes;
      const double x_extent;
      const double q0_extent;
      const uint q0_summands;
      const double jacobian_quadrature_factor;
      const JSONValue json;

      std::unique_ptr<DiFfRG::IntegratorAngleFiniteTq0TBB<4, double, etaQ_kernel<__REGULATOR__>>> integrator;
      std::unique_ptr<DiFfRG::IntegratorAngleFiniteTq0TBB<4, autodiff::real, etaQ_kernel<__REGULATOR__>>> integrator_AD;
    };
  } // namespace Flows
} // namespace DiFfRG