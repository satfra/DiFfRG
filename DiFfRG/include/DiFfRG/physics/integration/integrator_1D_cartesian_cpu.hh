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
   * @brief Integration of an arbitrary 1D function from qx_min to qx_max using TBB.
   *
   * @tparam NT The numerical type of the result.
   * @tparam KERNEL The kernel to integrate.
   */
  template <typename NT, typename KERNEL> class Integrator1DCartesianTBB
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    /**
     * @brief Construct a new Integrator1DCartesianTBB object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_size The number of grid points in x direction.
     * @param x_extent The extent of the x integration range. This argument is not used, but kept for compatibility with
     * flow classes.
     * @param qx_min The minimum value of the x integration range.
     * @param qx_max The maximum value of the x integration range.
     */
    Integrator1DCartesianTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const ctype x_extent = 0., const uint max_block_size = 0, const ctype qx_min = -M_PI,
                             const ctype qx_max = M_PI)
        : Integrator1DCartesianTBB(quadrature_provider, grid_size[0], x_extent, qx_min, qx_max)
    {
    }

    /**
     * @brief Construct a new Integrator1DCartesianTBB object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_size The number of grid points in x direction.
     * @param x_extent This argument is not used, but kept for compatibility with flow classes.
     * @param max_block_size The maximum block size to use on GPU. This argument is not used, but kept for
     * compatibility.
     * @param qx_min The minimum value of the x integration range.
     * @param qx_max The maximum value of the x integration range.
     */
    Integrator1DCartesianTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const JSONValue &json)
        : Integrator1DCartesianTBB(quadrature_provider, grid_size[0], 0.,
                                   json.get_double("/discretization/integration/qx_min", -M_PI),
                                   json.get_double("/discretization/integration/qx_max", M_PI))
    {
    }

    /**
     * @brief Construct a new Integrator1DCartesianTBB object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_size The number of grid points in x direction.
     * @param x_extent This argument is not used, but kept for compatibility with flow classes.
     * @param qx_min The minimum value of the x integration range.
     * @param qx_max The maximum value of the x integration range.
     */
    Integrator1DCartesianTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const ctype x_extent, const JSONValue &json)
        : Integrator1DCartesianTBB(quadrature_provider, grid_size[0], x_extent,
                                   json.get_double("/discretization/integration/qx_min", -M_PI),
                                   json.get_double("/discretization/integration/qx_max", M_PI))
    {
    }

    /**
     * @brief Construct a new Integrator1DCartesianTBB object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_size The number of grid points in x direction.
     * @param x_extent This argument is not used, but kept for compatibility with flow classes.
     * @param qx_min The minimum value of the x integration range.
     * @param qx_max The maximum value of the x integration range.
     */
    Integrator1DCartesianTBB(QuadratureProvider &quadrature_provider, const uint grid_size, const ctype x_extent = 0.,
                             const ctype qx_min = -M_PI, const ctype qx_max = M_PI)
        : grid_size(grid_size), x_quadrature_p(quadrature_provider.get_points<ctype>(grid_size)),
          x_quadrature_w(quadrature_provider.get_weights<ctype>(grid_size))
    {
      this->qx_min = qx_min;
      this->qx_extent = qx_max - qx_min;
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
     * @brief Set the maximum value of the qx integration range.
     */
    void set_qx_max(const ctype qx_max) { this->qx_extent = qx_max - qx_min; }

    /**
     * @brief Get the integration result.
     *
     * @tparam T The types of the additional arguments to the kernel.
     * @param k The current RG scale.
     * @param t The additional arguments of the kernel.
     * @return NT The result of the integration.
     */
    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      constexpr int d = 1;
      constexpr ctype int_element = powr<-d>(2 * (ctype)M_PI); // fourier factor

      return KERNEL::constant(k, t...) + tbb::parallel_reduce(
                                             tbb::blocked_range<uint>(0, grid_size), NT(0),
                                             [&](const tbb::blocked_range<uint> &r, NT value) -> NT {
                                               for (uint idx_x = r.begin(); idx_x != r.end(); ++idx_x) {
                                                 const ctype q = qx_min + qx_extent * x_quadrature_p[idx_x];
                                                 const ctype weight = qx_extent * x_quadrature_w[idx_x];
                                                 value += int_element * weight * KERNEL::kernel(q, k, t...);
                                               }
                                               return value;
                                             },
                                             [&](NT x, NT y) -> NT { return x + y; });
    }

    /**
     * @brief Get the integration result asynchronously.
     *
     * @tparam T The types of the additional arguments to the kernel.
     * @param k The current RG scale.
     * @param t The additional arguments of the kernel.
     * @return An std::future<NT> which returns the result of the integration.
     */
    template <typename... T> std::future<NT> request(const ctype k, const T &...t) const
    {
      return std::async(std::launch::deferred, [=, this]() { return get(k, t...); });
    }

  private:
    const uint grid_size;

    ctype qx_min = -M_PI;
    ctype qx_extent = 2. * M_PI;

    const std::vector<ctype> &x_quadrature_p;
    const std::vector<ctype> &x_quadrature_w;
  };
} // namespace DiFfRG