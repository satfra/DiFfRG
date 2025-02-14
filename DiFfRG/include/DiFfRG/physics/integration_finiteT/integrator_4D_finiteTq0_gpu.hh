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
  __global__ void gridreduce_4d_finiteTq0(NT *dest, const ctype *x_quadrature_p, const ctype *x_quadrature_w,
                                          const ctype *ang_quadrature_p, const ctype *ang_quadrature_w,
                                          const ctype *q0_quadrature_p, const ctype *q0_quadrature_w,
                                          const ctype x_extent, const ctype q0_extent, const uint q0_quadrature_size,
                                          const uint q0_summands, const ctype m_T, const ctype k, T... t)
  {
    constexpr uint d = 4;

    uint len_x = gridDim.x * blockDim.x;
    uint len_y = gridDim.y * blockDim.y;
    uint idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint idx_z = (blockIdx.z * blockDim.z) + threadIdx.z;
    uint idx = idx_z * len_x * len_y + idx_y * len_x + idx_x;

    const ctype q = k * sqrt(x_quadrature_p[idx_x] * x_extent);
    const ctype cos = 2 * (ang_quadrature_p[idx_y] - (ctype)0.5);
    const ctype phi = 2 * (ctype)M_PI * ang_quadrature_p[idx_z];
    const ctype weight =
        2 * (ctype)M_PI * ang_quadrature_w[idx_z] * 2 * ang_quadrature_w[idx_y] * x_quadrature_w[idx_x] * x_extent;

    NT res = 0;

    // integral
    const ctype int_element_int = (powr<d - 3>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                                  / powr<d>(2 * (ctype)M_PI);              // fourier factor

    const ctype integral_start = (2 * q0_summands * (ctype)M_PI * m_T);
    const ctype log_start = log(integral_start + (m_T == 0) * ctype(1e-3));
    const ctype log_ext = log(q0_extent / (integral_start + (m_T == 0) * ctype(1e-3)));

    for (uint idx_0 = 0; idx_0 < q0_quadrature_size; ++idx_0) {
      const ctype q0 = exp(log_start + log_ext * q0_quadrature_p[idx_0]) - (m_T == 0) * ctype(1e-3);
      const ctype m_weight = weight * (q0_quadrature_w[idx_0] * log_ext * q0);

      res += int_element_int * m_weight *
             (KERNEL::kernel(q, cos, phi, q0, k, t...) + KERNEL::kernel(q, cos, phi, -q0, k, t...));
    }

    // sum
    const ctype int_element_sum = m_T                                        // solid nd angle
                                  * (powr<d - 3>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                                  / powr<d - 1>(2 * (ctype)M_PI);            // fourier factor
    for (uint idx_0 = 0; idx_0 < q0_summands; ++idx_0) {
      const ctype q0 = 2 * (ctype)M_PI * m_T * idx_0;
      res += int_element_sum * weight *
             (idx_0 == 0 ? KERNEL::kernel(q, cos, phi, (ctype)0, k, t...)
                         : KERNEL::kernel(q, cos, phi, q0, k, t...) + KERNEL::kernel(q, cos, phi, -q0, k, t...));
    }

    dest[idx] = res;
  }

  template <typename NT, typename KERNEL> class Integrator4DFiniteTq0GPU
  {
  public:
    using ctype = typename get_type::ctype<NT>;

    Integrator4DFiniteTq0GPU(QuadratureProvider &quadrature_provider, const std::array<uint, 4> grid_sizes,
                             const ctype x_extent, const ctype q0_extent, const uint q0_summands, const JSONValue &json)
        : Integrator4DFiniteTq0GPU(quadrature_provider, grid_sizes, x_extent, q0_extent, q0_summands,
                                   json.get_double("/physical/T"), json.get_uint("/integration/cudathreadsperblock"))
    {
    }

    Integrator4DFiniteTq0GPU(QuadratureProvider &quadrature_provider, const std::array<uint, 4> _grid_sizes,
                             const ctype x_extent, const ctype q0_extent, const uint _q0_summands, const ctype T,
                             const uint max_block_size = 256)
        : quadrature_provider(quadrature_provider),
          grid_sizes({_grid_sizes[0], _grid_sizes[1], _grid_sizes[2], _grid_sizes[3] + _q0_summands}),
          device_data_size(grid_sizes[0] * grid_sizes[1] * grid_sizes[2]), x_extent(x_extent), q0_extent(q0_extent),
          original_q0_summands(_q0_summands),
          pool(rmm::mr::get_current_device_resource(), (device_data_size / 256 + 1) * 256)
    {
      if (grid_sizes[2] != grid_sizes[1])
        throw std::runtime_error("Grid sizes must be currently equal for all angular dimensions!");

      ptr_x_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[0]);
      ptr_x_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[0]);
      ptr_ang_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[1]);
      ptr_ang_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[1]);

      set_T(T);

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
    }

    void set_T(const ctype T)
    {
      m_T = T;
      if (is_close(T, 0.))
        q0_summands = 0;
      else
        q0_summands = original_q0_summands;
      ptr_q0_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[3] - q0_summands);
      ptr_q0_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[3] - q0_summands);
    }

    void set_q0_extent(const ctype val) { q0_extent = val; }

    Integrator4DFiniteTq0GPU(const Integrator4DFiniteTq0GPU &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes),
          device_data_size(other.device_data_size), ptr_x_quadrature_p(other.ptr_x_quadrature_p),
          ptr_x_quadrature_w(other.ptr_x_quadrature_w), ptr_ang_quadrature_p(other.ptr_ang_quadrature_p),
          ptr_ang_quadrature_w(other.ptr_ang_quadrature_w), ptr_q0_quadrature_p(other.ptr_q0_quadrature_p),
          ptr_q0_quadrature_w(other.ptr_q0_quadrature_w), x_extent(other.x_extent), q0_extent(other.q0_extent),
          original_q0_summands(other.original_q0_summands), q0_summands(other.q0_summands), m_T(other.m_T),
          pool(rmm::mr::get_current_device_resource(), (device_data_size / 256 + 1) * 256)
    {
      block_sizes = other.block_sizes;
      num_blocks = other.num_blocks;
      threads_per_block = other.threads_per_block;
    }

    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      const auto cuda_stream = cuda_stream_pool.get_stream();
      rmm::device_uvector<NT> device_data(device_data_size, cuda_stream, &pool);
      gridreduce_4d_finiteTq0<ctype, NT, KERNEL>
          <<<num_blocks, threads_per_block, 0, rmm::cuda_stream_per_thread.value()>>>(
              device_data.data(), ptr_x_quadrature_p, ptr_x_quadrature_w, ptr_ang_quadrature_p, ptr_ang_quadrature_w,
              ptr_q0_quadrature_p, ptr_q0_quadrature_w, x_extent, q0_extent, grid_sizes[3] - q0_summands, q0_summands,
              m_T, k, t...);
      check_cuda();
      return KERNEL::constant(k, t...) + thrust::reduce(thrust::cuda::par.on(rmm::cuda_stream_per_thread.value()),
                                                        device_data.begin(), device_data.end(), NT(0),
                                                        thrust::plus<NT>());
    }

    template <typename... T> std::future<NT> request(const ctype k, const T &...t) const
    {
      const auto cuda_stream = cuda_stream_pool.get_stream();
      std::shared_ptr<rmm::device_uvector<NT>> device_data =
          std::make_shared<rmm::device_uvector<NT>>(device_data_size, cuda_stream, &pool);
      gridreduce_4d_finiteTq0<ctype, NT, KERNEL><<<num_blocks, threads_per_block, 0, cuda_stream.value()>>>(
          (*device_data).data(), ptr_x_quadrature_p, ptr_x_quadrature_w, ptr_ang_quadrature_p, ptr_ang_quadrature_w,
          ptr_q0_quadrature_p, ptr_q0_quadrature_w, x_extent, q0_extent, grid_sizes[3] - q0_summands, q0_summands, m_T,
          k, t...);
      check_cuda();
      const NT constant = KERNEL::constant(k, t...);

      return std::async(std::launch::deferred, [=, this]() {
        return constant + thrust::reduce(thrust::cuda::par.on(cuda_stream.value()), (*device_data).begin(),
                                         (*device_data).end(), NT(0), thrust::plus<NT>());
      });
    }

  private:
    QuadratureProvider &quadrature_provider;

    const std::array<uint, 4> grid_sizes;
    std::array<uint, 3> block_sizes;

    const uint device_data_size;

    const ctype *ptr_x_quadrature_p;
    const ctype *ptr_x_quadrature_w;
    const ctype *ptr_ang_quadrature_p;
    const ctype *ptr_ang_quadrature_w;
    const ctype *ptr_q0_quadrature_p;
    const ctype *ptr_q0_quadrature_w;

    const ctype x_extent;
    ctype q0_extent;
    const uint original_q0_summands;
    uint q0_summands;
    ctype m_T;

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
  template <typename NT, typename KERNEL> class Integrator4DFiniteTq0GPU;
}

#else

#include <DiFfRG/physics/integration_finiteT/integrator_4D_finiteTq0_cpu.hh>

namespace DiFfRG
{
  template <typename NT, typename KERNEL> class Integrator4DFiniteTq0GPU : public Integrator4DFiniteTq0TBB<NT, KERNEL>
  {
  public:
    using ctype = typename get_type::ctype<NT>;

    Integrator4DFiniteTq0GPU(QuadratureProvider &quadrature_provider, const std::array<uint, 4> grid_sizes,
                             const ctype x_extent, const ctype q0_extent, const uint q0_summands, const ctype T,
                             const uint max_block_size = 256)
        : Integrator4DFiniteTq0TBB<NT, KERNEL>(quadrature_provider, grid_sizes, x_extent, q0_extent, q0_summands, T,
                                               max_block_size)
    {
    }
  };
} // namespace DiFfRG

#endif

#endif