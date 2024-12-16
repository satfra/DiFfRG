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
   * @brief Integrator for 2+1D integrals over p, x0 and an angle using TBB. Calculates
   * \f[
   *    \int dp dcos dx0 f(p, cos, x0) + c
   * \f]
   *
   * @tparam NT Numerical type.
   * @tparam KERNEL Kernel to integrate. Must have a static constant member function `constant` which supplies c and a
   * static member function `kernel` which supplies the integrand.
   */
  template <typename NT, typename KERNEL> class Integrator2Dpx0TBB
  {
  public:
    using ctype = typename get_type::ctype<NT>;

    /**
     * @brief Construct a new Integrator2Dpx0TBB object
     *
     * @param quadrature_provider QuadratureProvider object.
     * @param grid_sizes Grid sizes for the integrator.
     * @param x_extent Extent of the x-quadrature.
     * @param x0_extent Extent of the x0-quadrature.
     * @param json JSON object.
     */
    Integrator2Dpx0TBB(QuadratureProvider &quadrature_provider, const std::array<uint, 3> grid_sizes,
                       const ctype x_extent, const ctype x0_extent, const JSONValue &)
        : Integrator2Dpx0TBB(quadrature_provider, grid_sizes, x_extent, x0_extent)
    {
    }

    /**
     * @brief Construct a new Integrator2Dpx0TBB object
     *
     * @param quadrature_provider QuadratureProvider object.
     * @param grid_sizes Grid sizes for the integrator.
     * @param x_extent Extent of the x-quadrature.
     * @param x0_extent Extent of the x0-quadrature.
     * @param max_block_size unused.
     */
    Integrator2Dpx0TBB(QuadratureProvider &quadrature_provider, std::array<uint, 3> grid_sizes, const ctype x_extent,
                       const ctype x0_extent, const uint max_block_size = 0)
        : grid_sizes(grid_sizes), x_extent(x_extent), x0_extent(x0_extent),
          x_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[0])),
          x_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[0])),
          ang_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[1])),
          ang_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[1])),
          x0_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[2])),
          x0_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[2]))
    {
      (void)max_block_size;
    }

    /**
     * @brief Get the integral of the kernel.
     *
     * @tparam T Types of the arguments of the kernel.
     * @param k RG-scale.
     * @param t Parameters forwarded to the kernel.
     *
     * @return NT Integral of the kernel plus the constant part.
     */
    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      constexpr ctype S_2d = 2 * (ctype)M_PI;
      using std::sqrt;

      const auto constant = KERNEL::constant(k, t...);
      return constant +
             tbb::parallel_reduce(
                 tbb::blocked_range3d<uint, uint, uint>(0, grid_sizes[0], 0, grid_sizes[1], 0, grid_sizes[2]), NT(0.),
                 [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                   for (uint idx_x = r.pages().begin(); idx_x != r.pages().end(); ++idx_x) {
                     const ctype q = k * sqrt(x_quadrature_p[idx_x] * x_extent);
                     const ctype int_element = (1 / (ctype)2 * powr<2>(k)) // x = q^2 / k^2 integral
                                               * (k)                       // x0 = q0 / k integral
                                               * S_2d * (1 / (ctype)2)     // S_2 and divide out the cos integral
                                               / powr<3>(2 * (ctype)M_PI); // fourier factor
                     for (uint idx_y = r.rows().begin(); idx_y != r.rows().end(); ++idx_y) {
                       const ctype cos = 2 * (ang_quadrature_p[idx_y] - (ctype)0.5);
                       for (uint idx_z = r.cols().begin(); idx_z != r.cols().end(); ++idx_z) {
                         const ctype q0 = k * x0_quadrature_p[idx_z] * x0_extent;
                         const ctype weight = x0_quadrature_w[idx_z] * x0_extent * 2 * ang_quadrature_w[idx_y] *
                                              x_quadrature_w[idx_x] * x_extent;
                         value += int_element * weight *
                                  (KERNEL::kernel(q, cos, q0, k, t...) + KERNEL::kernel(q, cos, -q0, k, t...));
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
    const std::array<uint, 3> grid_sizes;

    const ctype x_extent;
    const ctype x0_extent;

    const std::vector<ctype> &x_quadrature_p;
    const std::vector<ctype> &x_quadrature_w;
    const std::vector<ctype> &ang_quadrature_p;
    const std::vector<ctype> &ang_quadrature_w;
    const std::vector<ctype> &x0_quadrature_p;
    const std::vector<ctype> &x0_quadrature_w;
  };
} // namespace DiFfRG