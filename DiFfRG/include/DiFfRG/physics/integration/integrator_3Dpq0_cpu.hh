#pragma once

// standard library
#include <future>

// external libraries
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

namespace DiFfRG
{
  template <typename NT, typename KERNEL> class Integrator3Dpq0TBB
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    Integrator3Dpq0TBB(QuadratureProvider &quadrature_provider, const std::array<uint, 4> grid_sizes,
                       const ctype x_extent, const ctype q0_extent, const JSONValue &)
        : Integrator3Dpq0TBB(quadrature_provider, grid_sizes, x_extent, q0_extent)
    {
    }

    Integrator3Dpq0TBB(QuadratureProvider &quadrature_provider, std::array<uint, 4> grid_sizes, const ctype x_extent,
                       const ctype q0_extent, const uint max_block_size = 0)
        : grid_sizes(grid_sizes), x_extent(x_extent), q0_extent(q0_extent),
          x_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[0])),
          x_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[0])),
          ang_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[1])),
          ang_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[1])),
          q0_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[2])),
          q0_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[2]))
    {
      (void)max_block_size;
      if (grid_sizes[2] != grid_sizes[1])
        throw std::runtime_error("Grid sizes must be currently equal for all angular dimensions!");
    }

    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      using std::sqrt;

      const auto constant = KERNEL::constant(k, t...);
      return constant +
             tbb::parallel_reduce(
                 tbb::blocked_range3d<uint, uint, uint>(0, grid_sizes[0], 0, grid_sizes[1], 0,
                                                        grid_sizes[2] * grid_sizes[3]),
                 NT(0.),
                 [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                   for (uint idx_x = r.pages().begin(); idx_x != r.pages().end(); ++idx_x) {
                     const ctype q = k * sqrt(x_quadrature_p[idx_x] * x_extent);
                     for (uint idx_y = r.rows().begin(); idx_y != r.rows().end(); ++idx_y) {
                       const ctype cos = 2 * (ang_quadrature_p[idx_y] - (ctype)0.5);
                       const ctype int_element = (powr<1>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                                                 / powr<4>(2 * (ctype)M_PI);          // fourier factor
                       for (uint idx_z = r.cols().begin(); idx_z != r.cols().end(); ++idx_z) {
                         const ctype phi = 2 * (ctype)M_PI * ang_quadrature_p[idx_z / grid_sizes[1]];
                         const ctype q0 = q0_quadrature_p[idx_z % grid_sizes[1]] * q0_extent;
                         const ctype weight = q0_quadrature_w[idx_z % grid_sizes[1]] * q0_extent * 2 * (ctype)M_PI *
                                              ang_quadrature_w[idx_z / grid_sizes[1]] * 2 * ang_quadrature_w[idx_y] *
                                              x_quadrature_w[idx_x] * x_extent;
                         value +=
                             int_element * weight *
                             (KERNEL::kernel(q, cos, phi, q0, k, t...) + KERNEL::kernel(q, cos, phi, -q0, k, t...));
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
    const ctype q0_extent;

    const std::vector<ctype> &x_quadrature_p;
    const std::vector<ctype> &x_quadrature_w;
    const std::vector<ctype> &ang_quadrature_p;
    const std::vector<ctype> &ang_quadrature_w;
    const std::vector<ctype> &q0_quadrature_p;
    const std::vector<ctype> &q0_quadrature_w;
  };
} // namespace DiFfRG