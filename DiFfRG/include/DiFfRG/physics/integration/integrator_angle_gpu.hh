#pragma once

#ifdef __CUDACC__

// standard library
#include <future>

// external libraries
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <thrust/reduce.h>

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

namespace DiFfRG
{
  template <typename ctype, int d, typename NT, typename KERNEL, typename... T>
  __global__ void gridreduce_angle(NT *dest, const ctype *x_quadrature_p, const ctype *x_quadrature_w,
                                   const ctype *ang_quadrature_p, const ctype *ang_quadrature_w, const ctype x_extent,
                                   const ctype k, T... t)
  {
    uint len_x = gridDim.x * blockDim.x;
    uint idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint idx = idx_y * len_x + idx_x;

    const ctype cos = 2 * (ang_quadrature_p[idx_y] - (ctype)0.5);
    const ctype weight = 2 * ang_quadrature_w[idx_y] * x_quadrature_w[idx_x] * x_extent;
    const ctype q = k * sqrt(x_quadrature_p[idx_x] * x_extent);
    constexpr ctype S_d = S_d_prec<ctype>(d);
    const ctype int_element = S_d                                        // solid nd angle
                              * (powr<d - 2>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                              * (1 / (ctype)2)                           // divide the cos integral out
                              / powr<d>(2 * (ctype)M_PI);                // fourier factor

    dest[idx] = int_element * weight * KERNEL::kernel(q, cos, k, t...);
  }

  /**
   * @brief GPU integrator for the integration of a function with one angle with CUDA. Calculates
   * \f[
   *    \int dp\, d\text{cos}\, \frac{1}{(2\pi)^d} f(p, \text{cos}, ...) + c
   * \f]
   * with \f$ p^2 \f$ bounded by \f$ \text{x_extent} * k^2 \f$.
   *
   * @tparam NT The numerical type of the result.
   * @tparam KERNEL The kernel to integrate.
   */
  template <int d, typename NT, typename KERNEL> class IntegratorAngleGPU
  {
    static_assert(d > 1, "IntegratorAngleGPU: d must be greater than 1, otherwise angles are not needed");

  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    IntegratorAngleGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 2> grid_sizes,
                       const ctype x_extent, const JSONValue &json)
        : IntegratorAngleGPU(quadrature_provider, grid_sizes, x_extent,
                             json.get_uint("/integration/cudathreadsperblock"))
    {
    }

    IntegratorAngleGPU(QuadratureProvider &quadrature_provider, std::array<uint, 2> grid_sizes, const ctype x_extent,
                       const uint max_block_size = 256)
        : grid_sizes(grid_sizes), device_data_size(grid_sizes[0] * grid_sizes[1]), x_extent(x_extent),
          pool(rmm::mr::get_current_device_resource(), (device_data_size / 256 + 1) * 256)
    {
      ptr_x_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[0]);
      ptr_x_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[0]);
      ptr_ang_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[1]);
      ptr_ang_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[1]);

      block_sizes = {max_block_size, max_block_size};
      // choose block sizes such that the size is both as close to max_block_size as possible and the individual sizes
      // are as close to each other as possible
      uint optimize_dim = 1;
      while (block_sizes[0] * block_sizes[1] > max_block_size || block_sizes[0] > grid_sizes[0] ||
             block_sizes[1] > grid_sizes[1]) {
        if (block_sizes[optimize_dim] > 1) block_sizes[optimize_dim]--;
        while (grid_sizes[optimize_dim] % block_sizes[optimize_dim] != 0)
          block_sizes[optimize_dim]--;
        optimize_dim = (optimize_dim + 1) % 2;
      }

      uint blocks1 = grid_sizes[0] / block_sizes[0];
      uint threads1 = block_sizes[0];
      uint blocks2 = grid_sizes[1] / block_sizes[1];
      uint threads2 = block_sizes[1];

      num_blocks = dim3(blocks1, blocks2);
      threads_per_block = dim3(threads1, threads2);
    }

    IntegratorAngleGPU(const IntegratorAngleGPU &other)
        : grid_sizes(other.grid_sizes), ptr_x_quadrature_p(other.ptr_x_quadrature_p),
          ptr_x_quadrature_w(other.ptr_x_quadrature_w), ptr_ang_quadrature_p(other.ptr_ang_quadrature_p),
          ptr_ang_quadrature_w(other.ptr_ang_quadrature_w), x_extent(other.x_extent),
          pool(rmm::mr::get_current_device_resource(), (device_data_size / 256 + 1) * 256)
    {
      device_data_size = other.device_data_size;
      block_sizes = other.block_sizes;
      num_blocks = other.num_blocks;
      threads_per_block = other.threads_per_block;
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
      const auto cuda_stream = cuda_stream_pool.get_stream();
      rmm::device_uvector<NT> device_data(device_data_size, cuda_stream.value(), &pool);
      gridreduce_angle<ctype, d, NT, KERNEL><<<num_blocks, threads_per_block, 0, cuda_stream.value()>>>(
          device_data.data(), ptr_x_quadrature_p, ptr_x_quadrature_w, ptr_ang_quadrature_p, ptr_ang_quadrature_w,
          x_extent, k, t...);
      check_cuda();
      return KERNEL::constant(k, t...) + thrust::reduce(thrust::cuda::par.on(cuda_stream.value()), device_data.begin(),
                                                        device_data.end(), NT(0.), thrust::plus<NT>());
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
      const auto cuda_stream = cuda_stream_pool.get_stream();
      std::shared_ptr<rmm::device_uvector<NT>> device_data =
          std::make_shared<rmm::device_uvector<NT>>(device_data_size, cuda_stream, &pool);
      gridreduce_angle<ctype, d, NT, KERNEL><<<num_blocks, threads_per_block, 0, cuda_stream.value()>>>(
          (*device_data).data(), ptr_x_quadrature_p, ptr_x_quadrature_w, ptr_ang_quadrature_p, ptr_ang_quadrature_w,
          x_extent, k, t...);
      check_cuda();
      const NT constant = KERNEL::constant(k, t...);

      return std::async(std::launch::deferred, [=, this]() {
        return constant + thrust::reduce(thrust::cuda::par.on(cuda_stream.value()), (*device_data).begin(),
                                         (*device_data).end(), NT(0.), thrust::plus<NT>());
      });
    }

  private:
    const std::array<uint, 2> grid_sizes;
    std::array<uint, 2> block_sizes;

    const uint device_data_size;

    const ctype *ptr_x_quadrature_p;
    const ctype *ptr_x_quadrature_w;
    const ctype *ptr_ang_quadrature_p;
    const ctype *ptr_ang_quadrature_w;

    const ctype x_extent;

    dim3 num_blocks;
    dim3 threads_per_block;

    using PoolMR = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
    mutable PoolMR pool;
    const rmm::cuda_stream_pool cuda_stream_pool;
  };
} // namespace DiFfRG

#else

#ifdef USE_CUDA

namespace DiFfRG
{
  template <uint d, typename NT, typename KERNEL> class IntegratorAngleGPU;
}

#else

#include <DiFfRG/physics/integration/integrator_angle_cpu.hh>

namespace DiFfRG
{
  template <uint d, typename NT, typename KERNEL> class IntegratorAngleGPU : public IntegratorAngleTBB<d, NT, KERNEL>
  {
  public:
    using ctype = typename get_type::ctype<NT>;

    IntegratorAngleGPU(QuadratureProvider &quadrature_provider, std::array<uint, 2> grid_sizes, const ctype x_extent,
                       const uint max_block_size = 256)
        : IntegratorAngleTBB<d, NT, KERNEL>(quadrature_provider, grid_sizes, x_extent)
    {
    }

    IntegratorAngleGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 2> grid_sizes,
                       const ctype x_extent, const JSONValue &)
        : IntegratorAngleTBB<d, NT, KERNEL>(quadrature_provider, grid_sizes, x_extent)
    {
    }
  };
} // namespace DiFfRG

#endif

#endif