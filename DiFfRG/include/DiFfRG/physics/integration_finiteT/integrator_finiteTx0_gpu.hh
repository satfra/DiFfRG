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
  template <typename ctype, int d, typename NT, typename KERNEL, typename... T>
  __global__ void gridreduce_1d_finiteTx0(NT *dest, const ctype *x_quadrature_p, const ctype *x_quadrature_w,
                                          const ctype *x0_quadrature_p, const ctype *x0_quadrature_w,
                                          const ctype x_extent, const ctype x0_extent, const uint x0_summands,
                                          const ctype m_T, const ctype k, T... t)
  {
    uint len_x = gridDim.x * blockDim.x;
    uint idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint idx = idx_y * len_x + idx_x;

    const ctype q = k * sqrt(x_quadrature_p[idx_x] * x_extent);
    const ctype S_dm1 = 2 * pow(M_PI, (d - 1.) / 2.) / tgamma((d - 1.) / 2.);

    NT res = 0;

    if (idx_y >= x0_summands) {
      const ctype integral_start = (2 * x0_summands * (ctype)M_PI * m_T) / k;
      const ctype log_start = log(integral_start + (m_T == 0) * ctype(1e-3));
      const ctype log_ext = log(x0_extent / (integral_start + (m_T == 0) * ctype(1e-3)));

      const ctype q0 = k * (exp(log_start + log_ext * x0_quadrature_p[idx_y - x0_summands]) - (m_T == 0) * ctype(1e-3));

      const ctype int_element = S_dm1                                      // solid nd angle
                                * (powr<d - 3>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                                * (k)                                      // x0 = q0 / k integral
                                / powr<d>(2 * (ctype)M_PI);                // fourier factor
      const ctype weight = x_quadrature_w[idx_x] * x_extent * (x0_quadrature_w[idx_y - x0_summands] * log_ext * q0 / k);

      res = int_element * weight * (KERNEL::kernel(q, q0, k, t...) + KERNEL::kernel(q, -q0, k, t...));
    } else {
      const ctype q0 = 2 * (ctype)M_PI * m_T * idx_y;
      const ctype int_element = m_T * S_dm1                                // temp * solid nd angle
                                * (powr<d - 3>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                                / powr<d - 1>(2 * (ctype)M_PI);            // fourier factor
      const ctype weight = x_quadrature_w[idx_x] * x_extent;
      res = int_element * weight *
            (idx_y == 0 ? KERNEL::kernel(q, (ctype)0, k, t...)
                        : KERNEL::kernel(q, q0, k, t...) + KERNEL::kernel(q, -q0, k, t...));
    }

    dest[idx] = res;
  }

  template <int d, typename NT, typename KERNEL> class IntegratorFiniteTx0GPU
  {
  public:
    using ctype = typename get_type::ctype<NT>;

    IntegratorFiniteTx0GPU(QuadratureProvider &quadrature_provider, const std::array<uint, 2> grid_sizes,
                           const ctype x_extent, const ctype x0_extent, const uint x0_summands, const JSONValue &json)
        : IntegratorFiniteTx0GPU(quadrature_provider, grid_sizes, x_extent, x0_extent, x0_summands,
                                 json.get_double("/physical/T"), json.get_uint("/integration/cudathreadsperblock"))
    {
    }

    IntegratorFiniteTx0GPU(QuadratureProvider &quadrature_provider, const std::array<uint, 2> _grid_sizes,
                           const ctype x_extent, const ctype x0_extent, const uint _x0_summands, const ctype T,
                           const uint max_block_size = 256)
        : quadrature_provider(quadrature_provider), grid_sizes{{_grid_sizes[0], _grid_sizes[1] + _x0_summands}},
          device_data_size(grid_sizes[0] * grid_sizes[1]), x_extent(x_extent), x0_extent(x0_extent),
          original_x0_summands(_x0_summands),
          pool(rmm::mr::get_current_device_resource(), (device_data_size / 256 + 1) * 256)
    {
      ptr_x_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[0]);
      ptr_x_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[0]);

      set_T(T);

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
    }

    void set_T(const ctype T)
    {
      m_T = T;
      if (is_close(T, 0.))
        x0_summands = 0;
      else
        x0_summands = original_x0_summands;
      ptr_x0_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[1] - x0_summands);
      ptr_x0_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[1] - x0_summands);
    }

    void set_x0_extent(const ctype val) { x0_extent = val; }

    IntegratorFiniteTx0GPU(const IntegratorFiniteTx0GPU &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes),
          device_data_size(other.device_data_size), ptr_x_quadrature_p(other.ptr_x_quadrature_p),
          ptr_x_quadrature_w(other.ptr_x_quadrature_w), ptr_x0_quadrature_p(other.ptr_x0_quadrature_p),
          ptr_x0_quadrature_w(other.ptr_x0_quadrature_w), x_extent(other.x_extent), x0_extent(other.x0_extent),
          original_x0_summands(other.original_x0_summands), x0_summands(other.x0_summands), m_T(other.m_T),
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
      gridreduce_1d_finiteTx0<ctype, d, NT, KERNEL><<<num_blocks, threads_per_block, 0, cuda_stream.value()>>>(
          device_data.data(), ptr_x_quadrature_p, ptr_x_quadrature_w, ptr_x0_quadrature_p, ptr_x0_quadrature_w,
          x_extent, x0_extent, x0_summands, m_T, k, t...);
      check_cuda();
      return KERNEL::constant(k, t...) + thrust::reduce(thrust::cuda::par.on(cuda_stream.value()), device_data.begin(),
                                                        device_data.end(), NT(0), thrust::plus<NT>());
    }

    template <typename... T> std::future<NT> request(const ctype k, const T &...t) const
    {
      const auto cuda_stream = cuda_stream_pool.get_stream();
      std::shared_ptr<rmm::device_uvector<NT>> device_data =
          std::make_shared<rmm::device_uvector<NT>>(device_data_size, cuda_stream, &pool);
      gridreduce_1d_finiteTx0<ctype, d, NT, KERNEL><<<num_blocks, threads_per_block, 0, cuda_stream.value()>>>(
          (*device_data).data(), ptr_x_quadrature_p, ptr_x_quadrature_w, ptr_x0_quadrature_p, ptr_x0_quadrature_w,
          x_extent, x0_extent, x0_summands, m_T, k, t...);
      check_cuda();
      const NT constant = KERNEL::constant(k, t...);

      return std::async(std::launch::deferred, [=, this]() {
        return constant + thrust::reduce(thrust::cuda::par.on(cuda_stream.value()), (*device_data).begin(),
                                         (*device_data).end(), NT(0), thrust::plus<NT>());
      });
    }

  private:
    QuadratureProvider &quadrature_provider;

    const std::array<uint, 2> grid_sizes;
    std::array<uint, 2> block_sizes;

    const uint device_data_size;

    const ctype *ptr_x_quadrature_p;
    const ctype *ptr_x_quadrature_w;
    const ctype *ptr_x0_quadrature_p;
    const ctype *ptr_x0_quadrature_w;

    const ctype x_extent;
    ctype x0_extent;
    const uint original_x0_summands;
    uint x0_summands;
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
  template <int d, typename NT, typename KERNEL> class IntegratorFiniteTx0GPU;
}

#else

#include <DiFfRG/physics/integration_finiteT/integrator_finiteTx0_cpu.hh>

namespace DiFfRG
{
  template <int d, typename NT, typename KERNEL>
  class IntegratorFiniteTx0GPU : public IntegratorFiniteTx0TBB<d, NT, KERNEL>
  {
  public:
    using ctype = typename get_type::ctype<NT>;

    IntegratorFiniteTx0GPU(QuadratureProvider &quadrature_provider, const std::array<uint, 2> _grid_sizes,
                           const ctype x_extent, const ctype x0_extent, const uint x0_summands, const ctype T,
                           const uint max_block_size = 256)
        : IntegratorFiniteTx0TBB<d, NT, KERNEL>(quadrature_provider, _grid_sizes, x_extent, x0_extent, x0_summands, T,
                                                max_block_size)
    {
    }
  };
} // namespace DiFfRG

#endif

#endif