#pragma once

// standard library
#include <future>

// external libraries
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/physics/integration/quadrature_provider.hh>

namespace DiFfRG
{
  template <int d, typename NT, typename KERNEL> class IntegratorTBB
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    IntegratorTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size, const ctype x_extent,
                  const uint max_block_size = 0)
        : IntegratorTBB(quadrature_provider, grid_size[0], x_extent, max_block_size)
    {
    }

    IntegratorTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size, const ctype x_extent,
                  const JSONValue &)
        : IntegratorTBB(quadrature_provider, grid_size[0], x_extent)
    {
    }

    IntegratorTBB(QuadratureProvider &quadrature_provider, const uint grid_size, const ctype x_extent,
                  const uint max_block_size = 0)
        : grid_size(grid_size), x_extent(x_extent), x_quadrature_p(quadrature_provider.get_points<ctype>(grid_size)),
          x_quadrature_w(quadrature_provider.get_weights<ctype>(grid_size))
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
      const ctype S_d = 2. * std::pow(M_PI, d / 2.) / std::tgammal(d / 2.);
      using std::sqrt;

      return KERNEL::constant(k, t...) +
             tbb::parallel_reduce(
                 tbb::blocked_range<uint>(0, grid_size), NT(0),
                 [&](const tbb::blocked_range<uint> &r, NT value) -> NT {
                   for (uint idx_x = r.begin(); idx_x != r.end(); ++idx_x) {
                     const ctype q = k * sqrt(x_quadrature_p[idx_x] * x_extent);
                     const ctype int_element = S_d                                        // solid nd angle
                                               * (powr<d - 2>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                                               / powr<d>(2 * (ctype)M_PI);                // fourier factor
                     const ctype weight = x_quadrature_w[idx_x] * x_extent;
                     value += int_element * weight * KERNEL::kernel(q, k, t...);
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
    const uint grid_size;

    const ctype x_extent;

    const std::vector<ctype> &x_quadrature_p;
    const std::vector<ctype> &x_quadrature_w;
  };
} // namespace DiFfRG