#pragma once

// standard library
#include <future>

// external libraries
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

namespace DiFfRG
{
  template <typename NT, typename KERNEL> class Integrator3DCartesianTBB
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    Integrator3DCartesianTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 3> grid_sizes,
                             const ctype x_extent, const JSONValue &json)
        : Integrator3DCartesianTBB(quadrature_provider, grid_sizes, x_extent, 0,
                                   json.get_double("/discretization/integration/qx_min", -M_PI),
                                   json.get_double("/discretization/integration/qy_min", -M_PI),
                                   json.get_double("/discretization/integration/qz_min", -M_PI),
                                   json.get_double("/discretization/integration/qx_max", M_PI),
                                   json.get_double("/discretization/integration/qy_max", M_PI),
                                   json.get_double("/discretization/integration/qz_max", M_PI))
    {
    }

    Integrator3DCartesianTBB(QuadratureProvider &quadrature_provider, std::array<uint, 3> grid_sizes,
                             const ctype x_extent = 0., const uint max_block_size = 0, const ctype qx_min = -M_PI,
                             const ctype qy_min = -M_PI, const ctype qz_min = -M_PI, const ctype qx_max = M_PI,
                             const ctype qy_max = M_PI, const ctype qz_max = M_PI)
        : grid_sizes(grid_sizes), x_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[0])),
          x_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[0])),
          y_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[1])),
          y_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[1])),
          z_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[2])),
          z_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[2]))
    {
      (void)max_block_size;
      (void)x_extent;
      this->qx_min = qx_min;
      this->qy_min = qy_min;
      this->qz_min = qz_min;
      this->qx_extent = qx_max - qx_min;
      this->qy_extent = qy_max - qy_min;
      this->qz_extent = qz_max - qz_min;
    }

    /**
     * @brief Set the minimum value of the qx integration range.
     */
    void set_qx_min(const ctype qx_min)
    {
      this->qx_extent = this->qx_extent - qx_min + this->qx_min;
      this->qx_min = qx_min;
    }

    /**
     * @brief Set the minimum value of the qy integration range.
     */
    void set_qy_min(const ctype qy_min)
    {
      this->qy_extent = this->qy_extent - qy_min + this->qy_min;
      this->qy_min = qy_min;
    }

    /**
     * @brief Set the minimum value of the qz integration range.
     */
    void set_qz_min(const ctype qz_min)
    {
      this->qz_extent = this->qz_extent - qz_min + this->qz_min;
      this->qz_min = qz_min;
    }

    /**
     * @brief Set the maximum value of the qx integration range.
     */
    void set_qx_max(const ctype qx_max) { this->qx_extent = qx_max - qx_min; }

    /**
     * @brief Set the maximum value of the qy integration range.
     */
    void set_qy_max(const ctype qy_max) { this->qy_extent = qy_max - qy_min; }

    /**
     * @brief Set the maximum value of the qz integration range.
     */
    void set_qz_max(const ctype qz_max) { this->qz_extent = qz_max - qz_min; }

    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      constexpr int d = 3;
      constexpr ctype int_element = powr<-d>(2 * (ctype)M_PI); // fourier factor

      const auto constant = KERNEL::constant(k, t...);
      return constant +
             tbb::parallel_reduce(
                 tbb::blocked_range3d<uint, uint, uint>(0, grid_sizes[0], 0, grid_sizes[1], 0, grid_sizes[2]), NT(0.),
                 [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                   for (uint idx_x = r.pages().begin(); idx_x != r.pages().end(); ++idx_x) {
                     const ctype qx = qx_min + qx_extent * x_quadrature_p[idx_x];
                     for (uint idx_y = r.rows().begin(); idx_y != r.rows().end(); ++idx_y) {
                       const ctype qy = qy_min + qy_extent * y_quadrature_p[idx_y];
                       for (uint idx_z = r.cols().begin(); idx_z != r.cols().end(); ++idx_z) {
                         const ctype qz = qz_min + qz_extent * z_quadrature_p[idx_z];

                         const ctype weight = qz_extent * z_quadrature_w[idx_z] * qy_extent * y_quadrature_w[idx_y] *
                                              qx_extent * x_quadrature_w[idx_x];
                         value += int_element * weight * KERNEL::kernel(qx, qy, qz, k, t...);
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

    ctype qx_min = -M_PI;
    ctype qy_min = -M_PI;
    ctype qx_extent = 2 * M_PI;
    ctype qy_extent = 2 * M_PI;
    ctype qz_min = -M_PI;
    ctype qz_extent = 2 * M_PI;

    const std::vector<ctype> &x_quadrature_p;
    const std::vector<ctype> &x_quadrature_w;
    const std::vector<ctype> &y_quadrature_p;
    const std::vector<ctype> &y_quadrature_w;
    const std::vector<ctype> &z_quadrature_p;
    const std::vector<ctype> &z_quadrature_w;
  };
} // namespace DiFfRG