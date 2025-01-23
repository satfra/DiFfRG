#pragma once

#ifdef __CUDACC__

// standard library
#include <future>

// external libraries
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/physics/integration/quadrature_provider.hh>

namespace DiFfRG
{
  template <int d, typename NT, typename KERNEL> class IntegratorGPU
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    /**
     * @brief Custom functor for the thrust::transform_reduce function.
     *
     * @tparam T The types of the additional arguments to the kernel.
     */
    template <typename... T> struct functor {
    public:
      functor(const ctype *x_quadrature_p, const ctype *x_quadrature_w, const ctype x_extent, const ctype k, T... t)
          : x_quadrature_p(x_quadrature_p), x_quadrature_w(x_quadrature_w), x_extent(x_extent), k(k), t(t...)
      {
      }

      __device__ NT operator()(const uint idx) const
      {
        const ctype weight = x_quadrature_w[idx] * x_extent;
        const ctype q = k * sqrt(x_quadrature_p[idx] * x_extent);
        const ctype S_d = 2 * pow(M_PI, d / 2.) / tgamma(d / 2.);
        const ctype int_element = S_d                                 // solid nd angle
                                  * (powr<d - 2>(q) / 2 * powr<2>(k)) // x = p^2 / k^2 integral
                                  / powr<d>(2 * (ctype)M_PI);         // fourier factor

        const NT res = std::apply([&](auto &&...args) { return KERNEL::kernel(q, k, args...); }, t);
        return int_element * res * weight;
      }

    private:
      const ctype *x_quadrature_p;
      const ctype *x_quadrature_w;
      const ctype x_extent;
      const ctype k;
      const std::tuple<T...> t;
    };

    IntegratorGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size, const ctype x_extent,
                  const uint max_block_size = 256)
        : IntegratorGPU(quadrature_provider, grid_size[0], x_extent, max_block_size)
    {
    }

    IntegratorGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size, const ctype x_extent,
                  const JSONValue &json)
        : IntegratorGPU(quadrature_provider, grid_size[0], x_extent, json.get_uint("/integration/cudathreadsperblock"))
    {
    }

    IntegratorGPU(QuadratureProvider &quadrature_provider, const uint grid_size, const ctype x_extent,
                  const uint max_block_size = 256)
        : grid_size(grid_size), x_extent(x_extent)
    {
      (void)max_block_size;
      ptr_x_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_size);
      ptr_x_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_size);
    }

    IntegratorGPU(const IntegratorGPU &other)
        : grid_size(other.grid_size), ptr_x_quadrature_p(other.ptr_x_quadrature_p),
          ptr_x_quadrature_w(other.ptr_x_quadrature_w), x_extent(other.x_extent)
    {
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
      return KERNEL::constant(k, t...) +
             thrust::transform_reduce(
                 thrust::cuda::par.on(rmm::cuda_stream_per_thread.value()), thrust::make_counting_iterator<uint>(0),
                 thrust::make_counting_iterator<uint>(grid_size),
                 functor<T...>(ptr_x_quadrature_p, ptr_x_quadrature_w, x_extent, k, t...), NT(0), thrust::plus<NT>());
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

    const ctype *ptr_x_quadrature_p;
    const ctype *ptr_x_quadrature_w;

    const ctype x_extent;
  };
} // namespace DiFfRG

#else

#ifdef USE_CUDA

namespace DiFfRG
{
  template <int d, typename NT, typename KERNEL> class IntegratorGPU;
}

#else

#include <DiFfRG/physics/integration/integrator_cpu.hh>

namespace DiFfRG
{
  template <int d, typename NT, typename KERNEL> class IntegratorGPU : public IntegratorTBB<d, NT, KERNEL>
  {
  public:
    using ctype = typename get_type::ctype<NT>;

    IntegratorGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size, const ctype x_extent,
                  const uint max_block_size = 256)
        : IntegratorTBB<d, NT, KERNEL>(quadrature_provider, grid_size, x_extent)
    {
      (void)max_block_size;
    }

    IntegratorGPU(QuadratureProvider &quadrature_provider, const uint grid_size, const ctype x_extent,
                  const uint max_block_size = 256)
        : IntegratorTBB<d, NT, KERNEL>(quadrature_provider, grid_size, x_extent)
    {
      (void)max_block_size;
    }

    IntegratorGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size, const ctype x_extent,
                  const JSONValue &)
        : IntegratorTBB<d, NT, KERNEL>(quadrature_provider, grid_size, x_extent)
    {
    }
  };
} // namespace DiFfRG

#endif

#endif