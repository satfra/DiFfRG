#pragma once

template <typename REG> class V_kernel;
#include "../def.hh"

#include <future>
#include <memory>

namespace DiFfRG
{
  namespace Flows
  {
    class V_integrator
    {
    public:
      V_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 1> grid_sizes, const double x_extent,
                   const JSONValue &json);
      V_integrator(const V_integrator &other);
      ~V_integrator();

      template <typename NT, typename... T> std::future<NT> request(T &&...t)
      {
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, autodiff::real>,
                      "Unknown type requested of V_integrator::request");
        if constexpr (std::is_same_v<NT, double>)
          return request_CT(std::forward<T>(t)...);
        else if constexpr (std::is_same_v<NT, autodiff::real>)
          return request_AD(std::forward<T>(t)...);
      }

      template <typename NT, typename... T> NT get(T &&...t)
      {
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, autodiff::real>,
                      "Unknown type requested of V_integrator::request");
        if constexpr (std::is_same_v<NT, double>)
          return get_CT(std::forward<T>(t)...);
        else if constexpr (std::is_same_v<NT, autodiff::real>)
          return get_AD(std::forward<T>(t)...);
      }

    private:
      std::future<double> request_CT(const double k, const double p0f, const double p, const double Nc, const double Nf,
                                     const double T, const double muq, const double etaQ, const double etaPhi,
                                     const double hPhi, const double d1V, const double d2V, const double d3V,
                                     const double rhoPhi);
      double get_CT(const double k, const double p0f, const double p, const double Nc, const double Nf, const double T,
                    const double muq, const double etaQ, const double etaPhi, const double hPhi, const double d1V,
                    const double d2V, const double d3V, const double rhoPhi);
      std::future<autodiff::real> request_AD(const double k, const double p0f, const double p, const double Nc,
                                             const double Nf, const double T, const double muq,
                                             const autodiff::real etaQ, const autodiff::real etaPhi,
                                             const autodiff::real hPhi, const autodiff::real d1V,
                                             const autodiff::real d2V, const autodiff::real d3V, const double rhoPhi);
      autodiff::real get_AD(const double k, const double p0f, const double p, const double Nc, const double Nf,
                            const double T, const double muq, const autodiff::real etaQ, const autodiff::real etaPhi,
                            const autodiff::real hPhi, const autodiff::real d1V, const autodiff::real d2V,
                            const autodiff::real d3V, const double rhoPhi);

      QuadratureProvider &quadrature_provider;
      const std::array<uint, 1> grid_sizes;
      std::array<uint, 1> jac_grid_sizes;
      const double x_extent;
      const double jacobian_quadrature_factor;
      const JSONValue json;

      std::unique_ptr<DiFfRG::IntegratorTBB<3, double, V_kernel<__REGULATOR__>>> integrator;
      std::unique_ptr<DiFfRG::IntegratorTBB<3, autodiff::real, V_kernel<__REGULATOR__>>> integrator_AD;
    };
  } // namespace Flows
} // namespace DiFfRG