#pragma once

// standard library
#include <future>

// external libraries
#include <qmc/qmc.hpp>

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/physics/integration/quadrature_provider.hh>

namespace DiFfRG
{
  /**
   * @brief GPU integrator for the integration of a function with one angle with quasi-Monte-Carlo. Calculates
   * \f[
   *    \int dp\, d\text{cos}\, \frac{1}{(2\pi)^d} f(p, \text{cos}, ...) + c
   * \f]
   * with \f$ p^2 \f$ bounded by \f$ \text{x_extent} * k^2 \f$.
   *
   * @tparam NT The numerical type of the result.
   * @tparam KERNEL The kernel to integrate.
   */
  template <int d, typename NT, typename KERNEL> class IntegratorAngleQMC
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    template <typename... Args> class Functor
    {
    public:
      static constexpr unsigned long long int number_of_integration_variables = 2;

      Functor(const ctype x_extent, const ctype k, const Args &...args) : x_extent(x_extent), k(k), args(args...) {}

      __host__ __device__ NT operator()(ctype *x) const
      {
        const ctype q = k * sqrt(x[0] * x_extent);
        const ctype cos = 2 * (x[1] - (ctype)0.5);
        const ctype weight = 2 * x_extent;
        const ctype S_d = 2 * pow(M_PI, d / 2.) / tgamma(d / 2.);
        const ctype int_element = S_d                                        // solid nd angle
                                  * (powr<d - 2>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                                  * (1 / (ctype)2)                           // divide the cos integral out
                                  / powr<d>(2 * (ctype)M_PI);                // fourier factor

        return std::apply([&](auto &&...args) { return int_element * weight * KERNEL::kernel(q, cos, k, args...); },
                          args);
      }

    private:
      const ctype x_extent;
      const ctype k;
      std::tuple<Args...> args;
    };

    IntegratorAngleQMC(QuadratureProvider &quadrature_provider, const std::array<uint, 2> grid_size,
                       const ctype x_extent, const JSONValue &json)
        : IntegratorAngleQMC(x_extent, json)
    {
    }

    IntegratorAngleQMC(const ctype x_extent, const double rel_tol = 1e-3, const double abs_tol = 1e-14,
                       const uint maxeval = 100000)
        : x_extent(x_extent)
    {
      integrator.verbosity = 0;
      integrator.epsrel = rel_tol;
      integrator.epsabs = abs_tol;
      integrator.maxeval = maxeval;
      integrator.minm = 32;
      integrator.minn = 512;
      integrator.cudablocks = 64;
      integrator.cudathreadsperblock = 32;
    }

    IntegratorAngleQMC(const ctype x_extent, const JSONValue &json) : x_extent(x_extent)
    {
      integrator.verbosity = 0;
      integrator.epsrel = json.get_double("/integration/rel_tol");
      integrator.epsabs = json.get_double("/integration/abs_tol");
      integrator.maxeval = json.get_int("/integration/max_eval");
      try {
        integrator.minm = json.get_int("/integration/minm");
        integrator.minn = json.get_int("/integration/minn");
        integrator.cudablocks = json.get_int("/integration/cudablocks");
        integrator.cudathreadsperblock = json.get_int("/integration/cudathreadsperblock");
      } catch (const std::exception &e) {
        spdlog::get("log")->warn("Please provide all integrator parameters in the json file. Using default values.");
        spdlog::get("log")->warn(
            "QMC integrator parameters: minm = 32, minn = 512, cudablocks = 64, cudathreadsperblock = 32");
        integrator.minm = 32;
        integrator.minn = 512;
        integrator.cudablocks = 64;
        integrator.cudathreadsperblock = 32;
      }
    }

    /**
     * @brief Get the integral of the kernel.
     *
     * @tparam T Types of the parameters for the kernel.
     * @param k RG-scale.
     * @param t Parameters forwarded to the kernel.
     *
     * @return NT Integral of the kernel plus the constant part.
     *
     */
    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      Functor<T...> functor(x_extent, k, t...);
      integrators::result<NT> result = integrator.integrate(functor);

      const NT constant = KERNEL::constant(k, t...);

      return constant + result.integral;
    }

    /**
     * @brief Request a future for the integral of the kernel.
     *
     * @tparam T Types of the parameters for the kernel.
     * @param k RG-scale.
     * @param t Parameters forwarded to the kernel.
     *
     * @return std::future<NT> future holding the integral of the kernel plus the constant part.
     *
     */
    template <typename... T> std::future<NT> request(const ctype k, const T &...t) const
    {
      const NT constant = KERNEL::constant(k, t...);

      return std::async(std::launch::deferred, [=, this]() {
        auto m_integrator = integrator;
        Functor<T...> functor(x_extent, k, t...);
        return constant + m_integrator.integrate(functor).integral;
      });
    }

  private:
    mutable integrators::Qmc<NT, ctype, 2, integrators::transforms::None::type> integrator;
    const ctype x_extent;
  };
} // namespace DiFfRG