#pragma once

// standard library
#include <future>

// external libraries
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

namespace DiFfRG
{
  template <int d, typename NT, typename KERNEL> class IntegratorFiniteTq0TBB
  {
    static_assert(d >= 2, "dimension must be at least 2");

  public:
    using ctype = typename get_type::ctype<NT>;

    IntegratorFiniteTq0TBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_sizes,
                           const ctype x_extent, const JSONValue &json)
        : IntegratorFiniteTq0TBB(quadrature_provider, grid_sizes, x_extent, json.get_double("/physical/T"))
    {
    }

    IntegratorFiniteTq0TBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> _grid_sizes,
                           const ctype x_extent, const ctype T, const uint max_block_size = 0)
        : quadrature_provider(quadrature_provider), grid_sizes{{_grid_sizes[0]}}, x_extent(x_extent), m_T(T)
    {
      ptr_x_quadrature_p = quadrature_provider.get_points<ctype>(grid_sizes[0]).data();
      ptr_x_quadrature_w = quadrature_provider.get_weights<ctype>(grid_sizes[0]).data();

      set_T(T);

      (void)max_block_size;
    }

    void set_T(const ctype T, const ctype typical_E = 0)
    {
      m_T = T;
      m_E = is_close(typical_E, 0.) ? m_T : typical_E;

      matsubara_size = quadrature_provider.get_matsubara_points<ctype>(m_T, m_E).size();
      ptr_matsubara_quadrature_p = quadrature_provider.get_matsubara_points<ctype>(m_T, m_E).data();
      ptr_matsubara_quadrature_w = quadrature_provider.get_matsubara_weights<ctype>(m_T, m_E).data();
    }

    IntegratorFiniteTq0TBB(const IntegratorFiniteTq0TBB &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes),
          ptr_x_quadrature_p(other.ptr_x_quadrature_p), ptr_x_quadrature_w(other.ptr_x_quadrature_w),
          ptr_matsubara_quadrature_p(other.ptr_matsubara_quadrature_p),
          ptr_matsubara_quadrature_w(other.ptr_matsubara_quadrature_w), x_extent(other.x_extent),
          matsubara_size(other.matsubara_size), m_T(other.m_T), m_E(other.m_E)
    {
    }

    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      const ctype S_dm1 = 2 * std::pow(M_PI, (d - 1) / 2.) / std::tgamma((d - 1) / 2.);
      using std::sqrt, std::exp, std::log;

      const auto constant = KERNEL::constant(k, t...);
      return constant +
             tbb::parallel_reduce(
                 tbb::blocked_range2d<uint, uint>(0, grid_sizes[0], 0, matsubara_size), NT(0),
                 [&](const tbb::blocked_range2d<uint, uint> &r, NT value) -> NT {
                   for (uint idx_x = r.rows().begin(); idx_x != r.rows().end(); ++idx_x) {
                     const ctype q = k * sqrt(ptr_x_quadrature_p[idx_x] * x_extent);

                     const ctype x_weight = ptr_x_quadrature_w[idx_x] * x_extent;
                     const ctype int_element = S_dm1                                      // solid nd angle
                                               * (powr<d - 3>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                                               / powr<d - 1>(2 * (ctype)M_PI);            // fourier factor

                     for (uint idx_y = r.cols().begin(); idx_y != r.cols().end(); ++idx_y) {
                       // integral part
                       const ctype q0 = ptr_matsubara_quadrature_p[idx_y];
                       const ctype weight = x_weight * ptr_matsubara_quadrature_w[idx_y];

                       value +=
                           int_element * weight * (KERNEL::kernel(q, q0, k, t...) + KERNEL::kernel(q, -q0, k, t...));
                       if (idx_y == 0) value += int_element * x_weight * m_T * KERNEL::kernel(q, (ctype)0, k, t...);
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

    const std::array<uint, 2> grid_sizes;

    const ctype x_extent;
    ctype m_T, m_E;

    const ctype *ptr_x_quadrature_p;
    const ctype *ptr_x_quadrature_w;
    const ctype *ptr_matsubara_quadrature_p;
    const ctype *ptr_matsubara_quadrature_w;
    uint matsubara_size;
  };
} // namespace DiFfRG