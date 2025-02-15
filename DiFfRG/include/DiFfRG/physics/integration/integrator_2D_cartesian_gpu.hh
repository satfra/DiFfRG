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
  /**
   * @brief GPU kernel for the integration of an arbitrary 2D function from qx_min to qx_max and qy_min to qy_max.
   *
   * @tparam ctype The numerical type of the integration points and weights.
   * @tparam NT The numerical type of the result.
   * @tparam KERNEL The kernel to integrate.
   */
  template <typename ctype, typename NT, typename KERNEL, typename... T>
  __global__ void gridreduce_2d_cartesian(NT *dest, const ctype *x_quadrature_p, const ctype *x_quadrature_w,
                                          const ctype *y_quadrature_p, const ctype *y_quadrature_w, const ctype qx_min,
                                          const ctype qy_min, const ctype qx_extent, const ctype qy_extent,
                                          const ctype k, T... t)
  {
    uint len_x = gridDim.x * blockDim.x;
    uint idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint idx = idx_y * len_x + idx_x;

    constexpr int d = 2;
    constexpr ctype int_element = powr<-d>(2 * (ctype)M_PI); // fourier factor

    const ctype qx = qx_min + qx_extent * x_quadrature_p[idx_x];
    const ctype qy = qy_min + qy_extent * y_quadrature_p[idx_y];
    const ctype weight = qy_extent * y_quadrature_w[idx_y] * qx_extent * x_quadrature_w[idx_x];

    dest[idx] = int_element * weight * KERNEL::kernel(qx, qy, k, t...);
  }

  /**
   * @brief Integration of an arbitrary 2D function from (qx_min, qy_min) to (qx_max, qy_max) using TBB.
   *
   * @tparam NT The numerical type of the result.
   * @tparam KERNEL The kernel to integrate.
   */
  template <typename NT, typename KERNEL> class Integrator2DCartesianGPU
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    /**
     * @brief Construct a new Integrator2DCartesianGPU object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_sizes The number of grid points in x and y direction.
     * @param x_extent This argument is not used, but kept for compatibility with flow classes.
     * @param json The JSON object containing the configuration.
     */
    Integrator2DCartesianGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 2> grid_sizes,
                             const ctype x_extent, const JSONValue &json)
        : Integrator2DCartesianGPU(quadrature_provider, grid_sizes, x_extent,
                                   json.get_uint("/integration/cudathreadsperblock"))
    {
    }

    /**
     * @brief Construct a new Integrator2DCartesianGPU object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_sizes The number of grid points in x and y direction.
     * @param x_extent This argument is not used, but kept for compatibility with flow classes.
     * @param max_block_size The maximum block size to use on GPU.
     * @param qy_min The minimum value of the y integration range.
     * @param qx_max The maximum value of the x integration range.
     * @param qy_max The maximum value of the y integration range.
     */
    Integrator2DCartesianGPU(QuadratureProvider &quadrature_provider, std::array<uint, 2> grid_sizes,
                             const ctype x_extent = 0., const uint max_block_size = 256, const ctype qx_min = -M_PI,
                             const ctype qy_min = -M_PI, const ctype qx_max = M_PI, const ctype qy_max = M_PI)
        : grid_sizes(grid_sizes), device_data_size(grid_sizes[0] * grid_sizes[1]),
          pool(rmm::mr::get_current_device_resource(), (device_data_size / 256 + 1) * 256)
    {
      ptr_x_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[0]);
      ptr_x_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[0]);
      ptr_y_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[1]);
      ptr_y_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[1]);

      block_sizes = {max_block_size, max_block_size};
      // choose block sizes such that the size is both as close to max_block_size as possible and the individual sizes
      // are as close to each other as possible
      uint optimize_dim = 1;
      while (block_sizes[0] * block_sizes[1] > max_block_size || block_sizes[0] > grid_sizes[0] ||
             block_sizes[1] > grid_sizes[1]) {
        block_sizes[optimize_dim]--;
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

      this->qx_min = qx_min;
      this->qy_min = qy_min;
      this->qx_extent = qx_max - qx_min;
      this->qy_extent = qy_max - qy_min;
    }

    /**
     * @brief Copy a Integrator2DCartesianGPU object
     *
     * @param other The other Integrator2DCartesianGPU object to copy.
     */
    Integrator2DCartesianGPU(const Integrator2DCartesianGPU &other)
        : grid_sizes(other.grid_sizes), ptr_x_quadrature_p(other.ptr_x_quadrature_p),
          ptr_x_quadrature_w(other.ptr_x_quadrature_w), ptr_y_quadrature_p(other.ptr_y_quadrature_p),
          ptr_y_quadrature_w(other.ptr_y_quadrature_w),
          pool(rmm::mr::get_current_device_resource(), (device_data_size / 256 + 1) * 256)
    {
      device_data_size = other.device_data_size;
      block_sizes = other.block_sizes;
      num_blocks = other.num_blocks;
      threads_per_block = other.threads_per_block;

      qx_min = other.qx_min;
      qy_min = other.qy_min;
      qx_extent = other.qx_extent;
      qy_extent = other.qy_extent;
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
     * @brief Get the result of the integration.
     *
     * @param k The current RG scale.
     * @param t The remaining arguments of the kernel.
     * @return The result of the integration.
     */
    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      const auto cuda_stream = cuda_stream_pool.get_stream();
      rmm::device_uvector<NT> device_data(device_data_size, cuda_stream.value(), &pool);
      gridreduce_2d_cartesian<ctype, NT, KERNEL><<<num_blocks, threads_per_block, 0, cuda_stream.value()>>>(
          device_data.data(), ptr_x_quadrature_p, ptr_x_quadrature_w, ptr_y_quadrature_p, ptr_y_quadrature_w, qx_min,
          qy_min, qx_extent, qy_extent, k, t...);
      check_cuda();
      return KERNEL::constant(k, t...) + thrust::reduce(thrust::cuda::par.on(cuda_stream.value()), device_data.begin(),
                                                        device_data.end(), NT(0.), thrust::plus<NT>());
    }

    /**
     * @brief Get the result of the integration asynchronously.
     *
     * @param k The current RG scale.
     * @param t The remaining arguments of the kernel.
     * @return An std::future<NT> which returns the result of the integration.
     */
    template <typename... T> std::future<NT> request(const ctype k, const T &...t) const
    {
      const auto cuda_stream = cuda_stream_pool.get_stream();
      std::shared_ptr<rmm::device_uvector<NT>> device_data =
          std::make_shared<rmm::device_uvector<NT>>(device_data_size, cuda_stream, &pool);
      gridreduce_2d_cartesian<ctype, NT, KERNEL><<<num_blocks, threads_per_block, 0, cuda_stream.value()>>>(
          (*device_data).data(), ptr_x_quadrature_p, ptr_x_quadrature_w, ptr_y_quadrature_p, ptr_y_quadrature_w, qx_min,
          qy_min, qx_extent, qy_extent, k, t...);
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

    ctype qx_min = -M_PI;
    ctype qy_min = -M_PI;
    ctype qx_extent = 2 * M_PI;
    ctype qy_extent = 2 * M_PI;

    const ctype *ptr_x_quadrature_p;
    const ctype *ptr_x_quadrature_w;
    const ctype *ptr_y_quadrature_p;
    const ctype *ptr_y_quadrature_w;

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
  template <typename NT, typename KERNEL> class Integrator2DCartesianGPU;
}

#else

#include <DiFfRG/physics/integration/integrator_2D_cartesian_cpu.hh>

namespace DiFfRG
{
  template <utypename NT, typename KERNEL> class Integrator2DCartesianGPU : public Integrator2DCartesianTBB<NT, KERNEL>
  {
  public:
    using ctype = typename get_type::ctype<NT>;

    Integrator2DCartesianGPU(QuadratureProvider &quadrature_provider, std::array<uint, 2> grid_sizes,
                             const ctype x_extent, const uint max_block_size = 256, const ctype qx_min = -M_PI,
                             const ctype qy_min = -M_PI, const ctype qx_max = M_PI, const ctype qy_max = M_PI)
        : Integrator2DCartesianTBB<NT, KERNEL>(quadrature_provider, grid_sizes, x_extent, max_block_size, qx_min,
                                               qy_min, qx_max, qy_max)
    {
    }

    Integrator2DCartesianGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 2> grid_sizes,
                             const ctype x_extent, const JSONValue &json)
        : Integrator2DCartesianTBB<NT, KERNEL>(quadrature_provider, grid_sizes, x_extent, json)
    {
    }
  };
} // namespace DiFfRG

#endif

#endif