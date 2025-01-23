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
#include <DiFfRG/physics/integration/quadrature_provider.hh>

namespace DiFfRG
{
  template <typename ctype, typename NT, typename KERNEL, typename... T>
  __global__ void gridreduce_4d_2ang(NT *dest, const ctype *x_quadrature_p, const ctype *x_quadrature_w,
                                     const ctype *ang_quadrature_p, const ctype *ang_quadrature_w, const ctype x_extent,
                                     const ctype k, T... t)
  {
    const uint len_x = gridDim.x * blockDim.x;
    const uint len_y = gridDim.y * blockDim.y;
    const uint idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const uint idx_z = (blockIdx.z * blockDim.z) + threadIdx.z;
    const uint idx = idx_z * len_x * len_y + idx_y * len_x + idx_x;

    const ctype q = k * sqrt(x_quadrature_p[idx_x] * x_extent);
    const ctype cos1 = 2 * (ang_quadrature_p[idx_y] - (ctype)0.5);
    const ctype int_element = (powr<2>(q) * (ctype)0.5 * powr<2>(k)) // x = p^2 / k^2 integral
                              * sqrt(1. - powr<2>(cos1))             // cos1 integral jacobian
                              * 2 * (ctype)M_PI                      // evaluated phi integral
                              / powr<4>(2 * (ctype)M_PI);            // fourier factor
    const ctype weight = 2 * ang_quadrature_w[idx_y]                 // cos1 weight
                         * x_quadrature_w[idx_x] * x_extent          // x weight
                         * 2 * ang_quadrature_w[idx_z];              // cos2 weight
    const ctype cos2 = 2 * (ang_quadrature_p[idx_z] - (ctype)0.5);

    dest[idx] = KERNEL::kernel(q, cos1, cos2, k, t...) // kernel
                * int_element * weight;                // weight and integration element
  }

  /**
   * @brief GPU integrator for the integration of a 4D function with two angles with CUDA. Calculates
   * \f[
   *    2\pi\,\int dp\, d\text{cos}_1\, d\text{cos}_2\, \frac{1}{(2\pi)^4} \sqrt{1-\text{cos}_1^2} f(p,
   * \text{cos}_1, \text{cos}_2, ...) + c
   * \f]
   * with \f$ p^2 \f$ bounded by \f$ \text{x_extent} * k^2 \f$.
   *
   * @tparam NT The numerical type of the result.
   * @tparam KERNEL The kernel to integrate.
   */
  template <typename NT, typename KERNEL> class Integrator4D2AngGPU
  {
    using PoolMR = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;

  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    Integrator4D2AngGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 3> grid_sizes,
                        const ctype x_extent, const JSONValue &json)
        : Integrator4D2AngGPU(quadrature_provider, grid_sizes, x_extent,
                              json.get_uint("/integration/cudathreadsperblock"))
    {
    }

    Integrator4D2AngGPU(QuadratureProvider &quadrature_provider, std::array<uint, 3> grid_sizes, const ctype x_extent,
                        const uint max_block_size = 256)
        : quadrature_provider(quadrature_provider), grid_sizes(grid_sizes),
          device_data_size(grid_sizes[0] * grid_sizes[1] * grid_sizes[2]), x_extent(x_extent)
    {
      cudaGetDeviceCount(&n_devices);
      if (n_devices == 0) throw std::runtime_error("No CUDA devices found!");

      for (int device = 0; device < n_devices; ++device) {
        const rmm::cuda_device_id device_id(device);
        pool.emplace_back(
            std::make_shared<PoolMR>(rmm::mr::get_per_device_resource(device_id), (device_data_size / 256 + 1) * 256));
      }

      if (grid_sizes[2] != grid_sizes[1])
        throw std::runtime_error("Grid sizes must be currently equal for all angular dimensions!");

      block_sizes = {max_block_size, max_block_size, max_block_size};
      // choose block sizes such that the size is both as close to max_block_size as possible and the individual sizes
      // are as close to each other as possible
      uint optimize_dim = 2;
      while (block_sizes[0] * block_sizes[1] * block_sizes[2] > max_block_size || block_sizes[0] > grid_sizes[0] ||
             block_sizes[1] > grid_sizes[1] || block_sizes[2] > grid_sizes[2]) {
        block_sizes[optimize_dim]--;
        while (grid_sizes[optimize_dim] % block_sizes[optimize_dim] != 0)
          block_sizes[optimize_dim]--;
        optimize_dim = (optimize_dim + 2) % 3;
      }

      uint blocks1 = grid_sizes[0] / block_sizes[0];
      uint threads1 = block_sizes[0];
      uint blocks2 = grid_sizes[1] / block_sizes[1];
      uint threads2 = block_sizes[1];
      uint blocks3 = grid_sizes[2] / block_sizes[2];
      uint threads3 = block_sizes[2];

      num_blocks = dim3(blocks1, blocks2, blocks3);
      threads_per_block = dim3(threads1, threads2, threads3);

      cudaSetDevice(0);
    }

    Integrator4D2AngGPU(const Integrator4D2AngGPU &other)
        : grid_sizes(other.grid_sizes), ptr_x_quadrature_p(other.ptr_x_quadrature_p),
          ptr_x_quadrature_w(other.ptr_x_quadrature_w), ptr_ang_quadrature_p(other.ptr_ang_quadrature_p),
          ptr_ang_quadrature_w(other.ptr_ang_quadrature_w), x_extent(other.x_extent), pool(other.pool)
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
      const int m_device = evaluations++ % n_devices;
      cudaSetDevice(m_device);

      const ctype *ptr_x_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[0], m_device);
      const ctype *ptr_x_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[0], m_device);
      const ctype *ptr_ang_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[1], m_device);
      const ctype *ptr_ang_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[1], m_device);

      const auto cuda_stream = cuda_stream_pool.get_stream();
      rmm::device_uvector<NT> device_data(device_data_size, cuda_stream, pool[m_device].get());
      gridreduce_4d_2ang<ctype, NT, KERNEL><<<num_blocks, threads_per_block, 0, cuda_stream.value()>>>(
          device_data.data(), ptr_x_quadrature_p, ptr_x_quadrature_w, ptr_ang_quadrature_p, ptr_ang_quadrature_w,
          x_extent, k, t...);
      check_cuda();
      return KERNEL::constant(k, t...) + thrust::reduce(thrust::cuda::par.on(cuda_stream.value()), device_data.begin(),
                                                        device_data.end(), NT(0.), thrust::plus<NT>());

      cudaSetDevice(0);
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
      const int m_device = evaluations++ % n_devices;
      cudaSetDevice(m_device);

      const ctype *ptr_x_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[0], m_device);
      const ctype *ptr_x_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[0], m_device);
      const ctype *ptr_ang_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[1], m_device);
      const ctype *ptr_ang_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[1], m_device);

      const auto cuda_stream = cuda_stream_pool.get_stream();
      std::shared_ptr<rmm::device_uvector<NT>> device_data =
          std::make_shared<rmm::device_uvector<NT>>(device_data_size, cuda_stream, pool[m_device].get());
      gridreduce_4d_2ang<ctype, NT, KERNEL><<<num_blocks, threads_per_block, 0, cuda_stream.value()>>>(
          (*device_data).data(), ptr_x_quadrature_p, ptr_x_quadrature_w, ptr_ang_quadrature_p, ptr_ang_quadrature_w,
          x_extent, k, t...);
      check_cuda();
      const NT constant = KERNEL::constant(k, t...);

      return std::async(std::launch::deferred, [=, this]() {
        cudaSetDevice(m_device);

        return constant + thrust::reduce(thrust::cuda::par.on(cuda_stream.value()), (*device_data).begin(),
                                         (*device_data).end(), NT(0.), thrust::plus<NT>());
      });

      cudaSetDevice(0);
    }

  private:
    QuadratureProvider &quadrature_provider;
    const std::array<uint, 3> grid_sizes;
    std::array<uint, 3> block_sizes;

    const uint device_data_size;

    const ctype *ptr_x_quadrature_p;
    const ctype *ptr_x_quadrature_w;
    const ctype *ptr_ang_quadrature_p;
    const ctype *ptr_ang_quadrature_w;

    const ctype x_extent;

    dim3 num_blocks;
    dim3 threads_per_block;

    int n_devices;
    mutable std::vector<std::shared_ptr<PoolMR>> pool;
    const rmm::cuda_stream_pool cuda_stream_pool;

    mutable std::atomic_ullong evaluations;
  };
} // namespace DiFfRG

#else

#ifdef USE_CUDA

namespace DiFfRG
{
  template <typename NT, typename KERNEL> class Integrator4D2AngGPU;
}

#else

#include <DiFfRG/physics/integration/integrator_4D_2ang_cpu.hh>

namespace DiFfRG
{
  template <typename NT, typename KERNEL> class Integrator4D2AngGPU : public Integrator4DTBB<NT, KERNEL>
  {
  public:
    using ctype = typename get_type::ctype<NT>;

    Integrator4D2AngGPU(QuadratureProvider &quadrature_provider, std::array<uint, 3> grid_sizes, const ctype x_extent,
                        const uint max_block_size = 256)
        : Integrator4D2AngTBB<NT, KERNEL>(quadrature_provider, grid_sizes, x_extent)
    {
    }

    Integrator4D2AngGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 3> grid_sizes,
                        const ctype x_extent, const JSONValue &)
        : Integrator4DTBB<NT, KERNEL>(quadrature_provider, grid_sizes, x_extent)
    {
    }
  };
} // namespace DiFfRG

#endif

#endif