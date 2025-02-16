#pragma once

// standard library
#include <future>

// external libraries
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

namespace DiFfRG
{
  template <typename NT, typename KERNEL> class Integrator4DFiniteTq0TBB
  {
  public:
    using ctype = typename get_type::ctype<NT>;

    Integrator4DFiniteTq0TBB(QuadratureProvider &quadrature_provider, const std::array<uint, 4> grid_sizes,
                             const ctype x_extent, const ctype q0_extent, const uint q0_summands, const JSONValue &json)
        : Integrator4DFiniteTq0TBB(quadrature_provider, grid_sizes, x_extent, json.get_double("/physical/T"),
                                   json.get_uint("/integration/cudathreadsperblock"))
    {
    }

    Integrator4DFiniteTq0TBB(QuadratureProvider &quadrature_provider, std::array<uint, 4> _grid_sizes,
                             const ctype x_extent, const ctype T, const uint max_block_size = 0)
        : quadrature_provider(quadrature_provider), grid_sizes{{_grid_sizes[0], _grid_sizes[1], _grid_sizes[2], 0}},
          x_extent(x_extent), manual_E(false)
    {
      ptr_x_quadrature_p = quadrature_provider.get_points(grid_sizes[0]).data();
      ptr_x_quadrature_w = quadrature_provider.get_weights(grid_sizes[0]).data();
      ptr_ang_quadrature_p = quadrature_provider.get_points(grid_sizes[1]).data();
      ptr_ang_quadrature_w = quadrature_provider.get_weights(grid_sizes[1]).data();

      set_T(T);

      (void)max_block_size;
    }

    /**
     * @brief Set the temperature and typical energy scale of the integrator and recompute the Matsubara quadrature
     * rule.
     *
     * @param T The temperature.
     * @param E A typical energy scale, which determines the number of nodes in the quadrature rule.
     */
    void set_T(const ctype T, const ctype E = 0)
    {
      m_T = T;
      // the default typical energy scale will default the matsubara size to 11.
      m_E = is_close(E, 0.) ? 10 * m_T : E;
      manual_E = !is_close(E, 0.);

      grid_sizes[3] = quadrature_provider.get_matsubara_points<ctype>(m_T, m_E).size();
      ptr_matsubara_quadrature_p = quadrature_provider.get_matsubara_points<ctype>(m_T, m_E).data();
      ptr_matsubara_quadrature_w = quadrature_provider.get_matsubara_weights<ctype>(m_T, m_E).data();
    }

    /**
     * @brief Set the typical energy scale of the integrator and recompute the Matsubara quadrature rule.
     *
     * @param E The typical energy scale.
     */
    void set_E(const ctype E) { set_T(m_T, E); }

    Integrator4DFiniteTq0TBB(const Integrator4DFiniteTq0TBB &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes),
          ptr_x_quadrature_p(other.ptr_x_quadrature_p), ptr_x_quadrature_w(other.ptr_x_quadrature_w),
          ptr_ang_quadrature_p(other.ptr_ang_quadrature_p), ptr_ang_quadrature_w(other.ptr_ang_quadrature_w),
          ptr_matsubara_quadrature_p(other.ptr_matsubara_quadrature_p),
          ptr_matsubara_quadrature_w(other.ptr_matsubara_quadrature_w), x_extent(other.x_extent), m_T(other.m_T),
          m_E(other.m_E), manual_E(other.manual_E)
    {
    }

    template <typename... T> NT get(const ctype k, const T &...t)
    {
      if (!manual_E && (std::abs(k - m_E) / std::max(k, m_E) > 2.5e-2)) {
        set_T(m_T, k);
        manual_E = false;
      }

      constexpr int d = 4;
      using std::sqrt, std::exp, std::log;

      const auto constant = KERNEL::constant(k, t...);
      return constant +
             tbb::parallel_reduce(
                 tbb::blocked_range3d<uint, uint, uint>(0, grid_sizes[0], 0, grid_sizes[1], 0, grid_sizes[2]), NT(0.),
                 [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                   for (uint idx_x = r.pages().begin(); idx_x != r.pages().end(); ++idx_x) {
                     const ctype q = k * sqrt(ptr_x_quadrature_p[idx_x] * x_extent);

                     const ctype int_element = (powr<d - 3>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                                               / powr<d - 1>(2 * (ctype)M_PI);          // fourier factor

                     for (uint idx_y = r.rows().begin(); idx_y != r.rows().end(); ++idx_y) {
                       const ctype cos = 2 * (ptr_ang_quadrature_p[idx_y] - (ctype)0.5);
                       for (uint idx_z = r.cols().begin(); idx_z != r.cols().end(); ++idx_z) {
                         const ctype phi = 2 * (ctype)M_PI * ptr_ang_quadrature_p[idx_z];

                         const ctype x_weight = 2 * (ctype)M_PI * ptr_ang_quadrature_w[idx_z] * 2 *
                                                ptr_ang_quadrature_w[idx_y] * ptr_x_quadrature_w[idx_x] * x_extent;

                         // summation
                         for (uint idx_0 = 0; idx_0 < grid_sizes[3]; ++idx_0) {
                           const ctype q0 = ptr_matsubara_quadrature_p[idx_0];
                           const ctype weight = x_weight * ptr_matsubara_quadrature_w[idx_0];
                           value +=
                               int_element * weight *
                               (KERNEL::kernel(q, cos, phi, q0, k, t...) + KERNEL::kernel(q, cos, phi, -q0, k, t...));
                           if (m_T > 0 && idx_0 == 0)
                             value += int_element * x_weight * m_T * KERNEL::kernel(q, cos, phi, (ctype)0, k, t...);
                         }
                       }
                     }
                   }
                   return value;
                 },
                 [&](NT x, NT y) -> NT { return x + y; });
    }

    template <typename... T> std::future<NT> request(const ctype k, const T &...t)
    {
      return std::async(std::launch::deferred, [=, this]() { return get(k, t...); });
    }

  private:
    QuadratureProvider &quadrature_provider;

    std::array<uint, 4> grid_sizes;

    const ctype x_extent;
    ctype m_T, m_E;
    bool manual_E;

    const ctype *ptr_x_quadrature_p;
    const ctype *ptr_x_quadrature_w;
    const ctype *ptr_ang_quadrature_p;
    const ctype *ptr_ang_quadrature_w;
    const ctype *ptr_matsubara_quadrature_p;
    const ctype *ptr_matsubara_quadrature_w;
  };
} // namespace DiFfRG