#pragma once

// standard library
#include <future>

// external libraries
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

namespace DiFfRG
{
  template <int d, typename NT, typename KERNEL> class IntegratorAngleFiniteTx0TBB
  {
    static_assert(d >= 2, "dimension must be at least 2");

  public:
    using ctype = typename get_type::ctype<NT>;

    IntegratorAngleFiniteTx0TBB(QuadratureProvider &quadrature_provider, const std::array<uint, 3> grid_sizes,
                                const ctype x_extent, const ctype x0_extent, const uint x0_summands,
                                const JSONValue &json)
        : IntegratorAngleFiniteTx0TBB(quadrature_provider, grid_sizes, x_extent, x0_extent, x0_summands,
                                      json.get_double("/physical/T"), json.get_uint("/integration/cudathreadsperblock"))
    {
    }

    IntegratorAngleFiniteTx0TBB(QuadratureProvider &quadrature_provider, std::array<uint, 3> _grid_sizes,
                                const ctype x_extent, const ctype x0_extent, const uint _x0_summands, const ctype T,
                                const uint max_block_size = 0)
        : quadrature_provider(quadrature_provider),
          grid_sizes{{_grid_sizes[0], _grid_sizes[1], _grid_sizes[2] + _x0_summands}}, x_extent(x_extent),
          x0_extent(x0_extent), original_x0_summands(_x0_summands)
    {
      ptr_x_quadrature_p = quadrature_provider.get_points<ctype>(grid_sizes[0]).data();
      ptr_x_quadrature_w = quadrature_provider.get_weights<ctype>(grid_sizes[0]).data();
      ptr_ang_quadrature_p = quadrature_provider.get_points<ctype>(grid_sizes[1]).data();
      ptr_ang_quadrature_w = quadrature_provider.get_weights<ctype>(grid_sizes[1]).data();

      set_T(T);

      (void)max_block_size;
    }

    void set_T(const ctype T)
    {
      m_T = T;
      if (is_close(T, 0.))
        x0_summands = 0;
      else
        x0_summands = original_x0_summands;
      ptr_x0_quadrature_p = quadrature_provider.get_points<ctype>(grid_sizes[2] - x0_summands).data();
      ptr_x0_quadrature_w = quadrature_provider.get_weights<ctype>(grid_sizes[2] - x0_summands).data();
    }

    void set_x0_extent(const ctype val) { x0_extent = val; }

    IntegratorAngleFiniteTx0TBB(const IntegratorAngleFiniteTx0TBB &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes),
          ptr_x_quadrature_p(other.ptr_x_quadrature_p), ptr_x_quadrature_w(other.ptr_x_quadrature_w),
          ptr_ang_quadrature_p(other.ptr_ang_quadrature_p), ptr_ang_quadrature_w(other.ptr_ang_quadrature_w),
          ptr_x0_quadrature_p(other.ptr_x0_quadrature_p), ptr_x0_quadrature_w(other.ptr_x0_quadrature_w),
          x_extent(other.x_extent), x0_extent(other.x0_extent), original_x0_summands(other.original_x0_summands),
          x0_summands(other.x0_summands), m_T(other.m_T)
    {
    }

    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      constexpr ctype S_dm1 = S_d_prec<ctype>(d - 1);
      using std::sqrt, std::exp, std::log;

      const ctype integral_start = (2 * x0_summands * (ctype)M_PI * m_T) / k;
      const ctype log_start = log(integral_start + (m_T == 0) * ctype(1e-3));
      const ctype log_ext = log(x0_extent / (integral_start + (m_T == 0) * ctype(1e-3)));

      const auto constant = KERNEL::constant(k, t...);
      return constant +
             tbb::parallel_reduce(
                 tbb::blocked_range3d<uint, uint, uint>(0, grid_sizes[0], 0, grid_sizes[1], 0, grid_sizes[2]), NT(0),
                 [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                   for (uint idx_x = r.pages().begin(); idx_x != r.pages().end(); ++idx_x) {
                     const ctype q = k * sqrt(ptr_x_quadrature_p[idx_x] * x_extent);
                     for (uint idx_y = r.rows().begin(); idx_y != r.rows().end(); ++idx_y) {
                       const ctype cos = 2 * (ptr_ang_quadrature_p[idx_y] - (ctype)0.5);
                       for (uint idx_z = r.cols().begin(); idx_z != r.cols().end(); ++idx_z) {
                         if (idx_z >= x0_summands) {
                           const ctype q0 = k * (exp(log_start + log_ext * ptr_x0_quadrature_p[idx_z - x0_summands]) -
                                                 (m_T == 0) * ctype(1e-3));

                           const ctype int_element = S_dm1 // solid nd angle
                                                     *
                                                     (powr<d - 3>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                                                     * (k)                                    // x0 = q0 / k integral
                                                     * (1 / (ctype)2)            // divide the cos integral out
                                                     / powr<d>(2 * (ctype)M_PI); // fourier factor
                           const ctype weight = 2 * ptr_ang_quadrature_w[idx_y] * ptr_x_quadrature_w[idx_x] * x_extent *
                                                (ptr_x0_quadrature_w[idx_z - x0_summands] * log_ext * q0 / k);

                           value += int_element * weight *
                                    (KERNEL::kernel(q, cos, q0, k, t...) + KERNEL::kernel(q, cos, -q0, k, t...));
                         } else {
                           const ctype q0 = 2 * (ctype)M_PI * m_T * idx_z;
                           const ctype int_element = m_T * S_dm1 // solid nd angle
                                                     *
                                                     (powr<d - 3>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                                                     * (1 / (ctype)2)                // divide the cos integral out
                                                     / powr<d - 1>(2 * (ctype)M_PI); // fourier factor
                           const ctype weight = 2 * ptr_ang_quadrature_w[idx_y] * ptr_x_quadrature_w[idx_x] * x_extent;
                           value += int_element * weight *
                                    (idx_z == 0
                                         ? KERNEL::kernel(q, cos, (ctype)0, k, t...)
                                         : KERNEL::kernel(q, cos, q0, k, t...) + KERNEL::kernel(q, cos, -q0, k, t...));
                         }
                       }
                     }
                   }
                   return value;
                 },
                 [&](NT x, NT y) -> NT { return x + y; });
    }

    template <typename... T> std::future<NT> request(const ctype k, const T &...t) const
    {
      return std::async(std::launch::deferred, [=, this]() { return get(k, t...); });
    }

  private:
    QuadratureProvider &quadrature_provider;

    const std::array<uint, 3> grid_sizes;

    const ctype x_extent;
    ctype x0_extent;
    const uint original_x0_summands;
    uint x0_summands;
    ctype m_T;

    const ctype *ptr_x_quadrature_p;
    const ctype *ptr_x_quadrature_w;
    const ctype *ptr_ang_quadrature_p;
    const ctype *ptr_ang_quadrature_w;
    const ctype *ptr_x0_quadrature_p;
    const ctype *ptr_x0_quadrature_w;
  };
} // namespace DiFfRG