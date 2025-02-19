#pragma once

// standard library
#include <future>

// external libraries
#include <openacc.h>
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

namespace DiFfRG
{
  template <typename NT, typename KERNEL> class Integrator4DOACC
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    Integrator4DOACC(QuadratureProvider &quadrature_provider, const std::array<uint, 4> grid_sizes,
                     const ctype x_extent, const JSONValue &)
        : Integrator4DOACC(quadrature_provider, grid_sizes, x_extent)
    {
    }

    Integrator4DOACC(QuadratureProvider &quadrature_provider, std::array<uint, 4> grid_sizes, const ctype x_extent,
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

      NT value = 0.;
#pragma acc parallel loop reduction(+ : value)
      for (uint idx_x = 0; idx_x < grid_sizes[0]; ++idx_x) {
        const ctype q = k * sqrt(x_quadrature_p[idx_x] * x_extent);

#pragma acc loop reduction(+ : value)
        for (uint idx_y = 0; idx_y < grid_sizes[1]; ++idx_y) {
          const ctype cos1 = 2 * (ang_quadrature_p1[idx_y] - (ctype)0.5);
          const ctype int_element = (powr<2>(q) * (ctype)0.5 * powr<2>(k)) // x = p^2 / k^2 integral
                                    * sqrt(1. - powr<2>(cos1))             // cos1 integral jacobian
                                    / powr<4>(2 * (ctype)M_PI);            // fourier factor

#pragma acc loop reduction(+ : value)
          for (uint cos2_idx = 0; cos2_idx < grid_sizes[2]; ++cos2_idx) {
            const ctype cos2 = 2 * (ang_quadrature_p2[cos2_idx] - (ctype)0.5);

            for (uint phi_idx = 0; phi_idx < grid_sizes[3]; ++phi_idx) {
              const ctype phi = 2 * (ctype)M_PI * ang_quadrature_p3[phi_idx];

              const ctype weight = 2 * (ctype)M_PI * ang_quadrature_w3[phi_idx] // phi weight
                                   * 2 * ang_quadrature_w2[cos2_idx]            // cos2 weight
                                   * 2 * ang_quadrature_w1[idx_y]               // cos1 weight
                                   * x_quadrature_w[idx_x] * x_extent;          // x weight

              value += int_element * weight * KERNEL::kernel(q, cos1, cos2, phi, k, t...);
            }
          }
        }
      }
      return constant + value;
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
