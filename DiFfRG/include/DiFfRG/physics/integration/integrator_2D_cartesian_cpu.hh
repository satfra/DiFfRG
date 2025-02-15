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
   * @brief Integration of an arbitrary 2D function from (qx_min, qy_min) to (qx_max, qy_max) using TBB.
   *
   * @tparam NT The numerical type of the result.
   * @tparam KERNEL The kernel to integrate.
   */
  template <typename NT, typename KERNEL> class Integrator2DCartesianTBB
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    /**
     * @brief Construct a new Integrator2DCartesianTBB object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_sizes The number of grid points in x and y direction.
     * @param x_extent The extent of the x integration range. This argument is not used, but kept for compatibility with
     * flow classes.
     * @param json The JSON object to read additional parameters from.
     */
    Integrator2DCartesianTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 2> grid_sizes,
                             const ctype x_extent, const JSONValue &json)
        : Integrator2DCartesianTBB(quadrature_provider, grid_sizes, x_extent, 0,
                                   json.get_double("/discretization/integration/qx_min", -M_PI),
                                   json.get_double("/discretization/integration/qy_min", -M_PI),
                                   json.get_double("/discretization/integration/qx_max", M_PI),
                                   json.get_double("/discretization/integration/qy_max", M_PI))
    {
    }

    /**
     * @brief Construct a new Integrator2DCartesianTBB object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_sizes The number of grid points in x and y direction.
     * @param x_extent The extent of the x integration range. This argument is not used, but kept for compatibility with
     * flow classes.
     * @param max_block_size The maximum block size to use on GPU. This argument is not used, but kept for
     * compatibility.
     * @param qx_min The minimum value of the x integration range.
     * @param qy_min The minimum value of the y integration range.
     * @param qx_max The maximum value of the x integration range.
     * @param qy_max The maximum value of the y integration range.
     */
    Integrator2DCartesianTBB(QuadratureProvider &quadrature_provider, std::array<uint, 2> grid_sizes,
                             const ctype x_extent, const uint max_block_size = 0, const ctype qx_min = -M_PI,
                             const ctype qy_min = -M_PI, const ctype qx_max = M_PI, const ctype qy_max = M_PI)
        : grid_sizes(grid_sizes), x_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[0])),
          x_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[0])),
          y_quadrature_p(quadrature_provider.get_points<ctype>(grid_sizes[1])),
          y_quadrature_w(quadrature_provider.get_weights<ctype>(grid_sizes[1]))
    {
      (void)max_block_size;
      this->qx_min = qx_min;
      this->qy_min = qy_min;
      this->qx_extent = qx_max - qx_min;
      this->qy_extent = qy_max - qy_min;
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
     * @brief Set the maximum value of the qx integration range.
     */
    void set_qx_max(const ctype qx_max) { this->qx_extent = qx_max - qx_min; }

    /**
     * @brief Set the maximum value of the qy integration range.
     */
    void set_qy_max(const ctype qy_max) { this->qy_extent = qy_max - qy_min; }

    /**
     * @brief Get the value of the integral.
     *
     * @param k The current RG scale.
     * @param t The additional parameters for the kernel.
     * @return The value of the integral.
     */
    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      constexpr int d = 2;
      constexpr ctype int_element = powr<-d>(2 * (ctype)M_PI); // fourier factor

      const auto constant = KERNEL::constant(k, t...);
      return constant + tbb::parallel_reduce(
                            tbb::blocked_range2d<uint, uint>(0, grid_sizes[0], 0, grid_sizes[1]), NT(0.),
                            [&](const tbb::blocked_range2d<uint, uint> &r, NT value) -> NT {
                              for (uint idx_x = r.rows().begin(); idx_x != r.rows().end(); ++idx_x) {
                                const ctype qx = qx_min + qx_extent * x_quadrature_p[idx_x];
                                for (uint idx_y = r.cols().begin(); idx_y != r.cols().end(); ++idx_y) {
                                  const ctype qy = qy_min + qy_extent * y_quadrature_p[idx_y];

                                  const ctype weight =
                                      qy_extent * y_quadrature_w[idx_y] * qx_extent * x_quadrature_w[idx_x];
                                  value += int_element * weight * KERNEL::kernel(qx, qy, k, t...);
                                }
                              }
                              return value;
                            },
                            [&](NT x, NT y) -> NT { return x + y; });
    }

    /**
     * @brief Get the value of the integral asynchronously.
     *
     * @param k The current RG scale.
     * @param t The additional parameters for the kernel.
     * @return An std::future<NT> which returns the value of the integral.
     */
    template <typename... T> std::future<NT> request(const ctype k, const T &...t) const
    {
      return std::async(std::launch::deferred, [=, this]() { return get(k, t...); });
    }

  private:
    const std::array<uint, 2> grid_sizes;

    ctype qx_min = -M_PI;
    ctype qy_min = -M_PI;
    ctype qx_extent = 2 * M_PI;
    ctype qy_extent = 2 * M_PI;

    const std::vector<ctype> &x_quadrature_p;
    const std::vector<ctype> &x_quadrature_w;
    const std::vector<ctype> &y_quadrature_p;
    const std::vector<ctype> &y_quadrature_w;
  };
} // namespace DiFfRG