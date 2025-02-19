#pragma once

// standard library
#include <future>

// external libraries
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

namespace DiFfRG
{
  /**
   * @brief Integrator for the integration of a 4D function with three angles with TBB. Calculates
   * \f[
   *    \int dp\, d\text{cos}_1\, d\text{cos}_2\, d\phi\, \frac{1}{(2\pi)^4} \sqrt{1-\text{cos}_1^2} f(p,
   * \text{cos}_1, \text{cos}_2, \phi, ...) + c
   * \f]
   * with \f$ p^2 \f$ bounded by \f$ \text{x_extent} * k^2 \f$.
   *
   * @tparam NT The numerical type of the result.
   * @tparam KERNEL The kernel to integrate.
   */
  template <typename NT, typename KERNEL> class Integrator4DTBB
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    Integrator4DTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 4> grid_sizes, const ctype x_extent,
                    const JSONValue &)
        : Integrator4DTBB(quadrature_provider, grid_sizes, x_extent)
    {
    }

    Integrator4DTBB(QuadratureProvider &quadrature_provider, std::array<uint, 4> grid_sizes, const ctype x_extent,
                    const uint max_block_size = 0)
        : grid_sizes(grid_sizes), x_extent(x_extent),
          x_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[0])),
          x_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[0])),
          ang_quadrature_p1(quadrature_provider.get_points<ctype>(grid_sizes[1])),
          ang_quadrature_w1(quadrature_provider.get_weights<ctype>(grid_sizes[1])),
          ang_quadrature_p2(quadrature_provider.get_points<ctype>(grid_sizes[2])),
          ang_quadrature_w2(quadrature_provider.get_weights<ctype>(grid_sizes[2])),
          ang_quadrature_p3(quadrature_provider.get_points<ctype>(grid_sizes[3])),
          ang_quadrature_w3(quadrature_provider.get_weights<ctype>(grid_sizes[3]))
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
      using std::sqrt;

      const auto constant = KERNEL::constant(k, t...);
      return constant + tbb::parallel_reduce(
                            tbb::blocked_range3d<uint, uint, uint>(0, grid_sizes[0], 0, grid_sizes[1], 0,
                                                                   grid_sizes[2] * grid_sizes[3]),
                            NT(0),
                            [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                              for (uint idx_x = r.pages().begin(); idx_x != r.pages().end(); ++idx_x) {
                                const ctype q = k * sqrt(x_quadrature_p[idx_x] * x_extent);
                                for (uint idx_y = r.rows().begin(); idx_y != r.rows().end(); ++idx_y) {
                                  const ctype cos1 = 2 * (ang_quadrature_p1[idx_y] - (ctype)0.5);
                                  const ctype int_element =
                                      (powr<2>(q) * (ctype)0.5 * powr<2>(k)) // x = p^2 / k^2 integral
                                      * sqrt(1. - powr<2>(cos1))             // cos1 integral jacobian
                                      / powr<4>(2 * (ctype)M_PI);            // fourier factor
                                  for (uint idx_z = r.cols().begin(); idx_z != r.cols().end(); ++idx_z) {
                                    const uint cos2_idx = idx_z / grid_sizes[2];
                                    const uint phi_idx = idx_z % grid_sizes[3];
                                    const ctype cos2 = 2 * (ang_quadrature_p2[cos2_idx] - (ctype)0.5);
                                    const ctype phi = 2 * (ctype)M_PI * ang_quadrature_p3[phi_idx];
                                    const ctype weight = 2 * (ctype)M_PI * ang_quadrature_w3[phi_idx] // phi weight
                                                         * 2 * ang_quadrature_w2[cos2_idx]            // cos2 weight
                                                         * 2 * ang_quadrature_w1[idx_y]               // cos1 weight
                                                         * x_quadrature_w[idx_x] * x_extent;          // x weight
                                    value += int_element * weight * KERNEL::kernel(q, cos1, cos2, phi, k, t...);
                                  }
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
    const std::array<uint, 4> grid_sizes;

    const ctype x_extent;

    const std::vector<ctype> &x_quadrature_p;
    const std::vector<ctype> &x_quadrature_w;
    const std::vector<ctype> &ang_quadrature_p1;
    const std::vector<ctype> &ang_quadrature_w1;
    const std::vector<ctype> &ang_quadrature_p2;
    const std::vector<ctype> &ang_quadrature_w2;
    const std::vector<ctype> &ang_quadrature_p3;
    const std::vector<ctype> &ang_quadrature_w3;
  };
} // namespace DiFfRG