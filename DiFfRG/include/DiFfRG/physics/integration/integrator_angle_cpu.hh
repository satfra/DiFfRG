#pragma once

// standard library
#include <future>

// external libraries
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/physics/integration/quadrature_provider.hh>

namespace DiFfRG
{
  /**
   * @brief Integrator for the integration of a function with one angle with TBB. Calculates
   * \f[
   *    \int dp\, d\text{cos}\, \frac{1}{(2\pi)^d} f(p, \text{cos}, ...) + c
   * \f]
   * with \f$ p^2 \f$ bounded by \f$ \text{x_extent} * k^2 \f$.
   *
   * @tparam NT The numerical type of the result.
   * @tparam KERNEL The kernel to integrate.
   */
  template <int d, typename NT, typename KERNEL> class IntegratorAngleTBB
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    IntegratorAngleTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 2> grid_sizes,
                       const ctype x_extent, const JSONValue &)
        : IntegratorAngleTBB(quadrature_provider, grid_sizes, x_extent)
    {
    }

    IntegratorAngleTBB(QuadratureProvider &quadrature_provider, std::array<uint, 2> grid_sizes, const ctype x_extent,
                       const uint max_block_size = 0)
        : grid_sizes(grid_sizes), x_extent(x_extent),
          x_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[0])),
          x_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[0])),
          ang_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[1])),
          ang_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[1]))
    {
      (void)max_block_size;
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
      const ctype S_d = 2 * std::pow(M_PI, d / 2.) / std::tgamma(d / 2.);
      using std::sqrt;

      const auto constant = KERNEL::constant(k, t...);
      return constant + tbb::parallel_reduce(
                            tbb::blocked_range2d<uint, uint>(0, grid_sizes[0], 0, grid_sizes[1]), NT(0.),
                            [&](const tbb::blocked_range2d<uint, uint> &r, NT value) -> NT {
                              for (uint idx_x = r.rows().begin(); idx_x != r.rows().end(); ++idx_x) {
                                const ctype q = k * sqrt(x_quadrature_p[idx_x] * x_extent);
                                for (uint idx_y = r.cols().begin(); idx_y != r.cols().end(); ++idx_y) {
                                  const ctype cos = 2 * (ang_quadrature_p[idx_y] - (ctype)0.5);
                                  const ctype int_element =
                                      S_d                                        // solid nd angle
                                      * (powr<d - 2>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                                      * (1 / (ctype)2)                           // divide the cos integral out
                                      / powr<d>(2 * (ctype)M_PI);                // fourier factor
                                  const ctype weight = 2 * ang_quadrature_w[idx_y] * x_quadrature_w[idx_x] * x_extent;
                                  value += int_element * weight * KERNEL::kernel(q, cos, k, t...);
                                }
                              }
                              return value;
                            },
                            [&](NT x, NT y) -> NT { return x + y; });
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
      return std::async(std::launch::deferred, [=, this]() { return get(k, t...); });
    }

  private:
    const std::array<uint, 2> grid_sizes;

    const ctype x_extent;

    const std::vector<ctype> &x_quadrature_p;
    const std::vector<ctype> &x_quadrature_w;
    const std::vector<ctype> &ang_quadrature_p;
    const std::vector<ctype> &ang_quadrature_w;
  };
} // namespace DiFfRG