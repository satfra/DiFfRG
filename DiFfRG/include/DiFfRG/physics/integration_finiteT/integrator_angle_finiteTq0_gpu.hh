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
  __global__ void gridreduce_angle_finiteTq0(NT *dest, const ctype *x_quadrature_p, const ctype *x_quadrature_w,
                                             const ctype *ang_quadrature_p, const ctype *ang_quadrature_w,
                                             const ctype *matsubara_quadrature_p, const ctype *matsubara_quadrature_w,
                                             const ctype x_extent, const ctype m_T, const ctype k, T... t)
  {
    uint len_x = gridDim.x * blockDim.x;
    uint len_y = gridDim.y * blockDim.y;
    uint idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint idx_z = (blockIdx.z * blockDim.z) + threadIdx.z;
    uint idx = idx_z * len_x * len_y + idx_y * len_x + idx_x;

    const ctype q = k * sqrt(x_quadrature_p[idx_x] * x_extent);
    const ctype cos = 2 * (ang_quadrature_p[idx_y] - (ctype)0.5);
    constexpr ctype S_dm1 = S_d_prec<ctype>(d - 1);

    const ctype x_weight = 2 * ang_quadrature_w[idx_y] * x_quadrature_w[idx_x] * x_extent;
    const ctype int_element = S_dm1                                      // solid nd angle
                              * (powr<d - 3>(q) / (ctype)2 * powr<2>(k)) // x = p^2 / k^2 integral
                              * (1. / (ctype)2)                          // divide the cos integral out
                              / powr<d - 1>(2 * (ctype)M_PI);            // fourier factor

    const ctype q0 = matsubara_quadrature_p[idx_z];
    const ctype weight = x_weight * matsubara_quadrature_w[idx_z];

    NT res = int_element * weight * (KERNEL::kernel(q, cos, q0, k, t...) + KERNEL::kernel(q, cos, -q0, k, t...));
    if (m_T > 0 && idx_z == 0) res += int_element * x_weight * m_T * KERNEL::kernel(q, cos, (ctype)0, k, t...);

    dest[idx] = res;
  }

  template <int d, typename NT, typename KERNEL> class IntegratorAngleFiniteTq0GPU
  {
    static_assert(d >= 3, "dimension must be at least 3");

  public:
    using ctype = typename get_type::ctype<NT>;

    IntegratorAngleFiniteTq0GPU(QuadratureProvider &quadrature_provider, const std::array<uint, 2> grid_sizes,
                                const ctype x_extent, const JSONValue &json)
        : IntegratorAngleFiniteTq0GPU(quadrature_provider, grid_sizes, x_extent, json.get_double("/physical/T"),
                                      json.get_uint("/integration/cudathreadsperblock"))
    {
    }

    IntegratorAngleFiniteTq0GPU(QuadratureProvider &quadrature_provider, const std::array<uint, 2> _grid_sizes,
                                const ctype x_extent, const ctype T, const uint max_block_size = 256)
        : quadrature_provider(quadrature_provider), grid_sizes({_grid_sizes[0], _grid_sizes[1], 32}),
          device_data_size(grid_sizes[0] * grid_sizes[1] * grid_sizes[2]), x_extent(x_extent),
          pool(rmm::mr::get_current_device_resource(), (device_data_size / 256 + 1) * 256),
          max_block_size(max_block_size), manual_E(false)
    {
      set_T(T);
    }

    void reinit()
    {
      device_data_size = grid_sizes[0] * grid_sizes[1] * grid_sizes[2];

      ptr_x_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[0]);
      ptr_x_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[0]);
      ptr_ang_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_sizes[1]);
      ptr_ang_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_sizes[1]);

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

    /**
     * @brief Set the temperature and typical energy scale of the integrator and recompute the Matsubara quadrature
     * rule.
     *
     * @param T The temperature.
     * @param E A typical energy scale, which determines the number of nodes in the quadrature rule.
     */
    void set_T(const ctype T, const ctype E = 0)
    {
      m_T = T;
      // the default typical energy scale will default the matsubara size to 11.
      m_E = is_close(E, 0.) ? 10 * m_T : E;
      manual_E = !is_close(E, 0.);

      const auto old_size = grid_sizes[2];
      grid_sizes[2] = quadrature_provider.get_matsubara_points<ctype>(m_T, m_E).size();

      if (old_size != grid_sizes[2]) {
        ptr_matsubara_quadrature_p = quadrature_provider.get_device_matsubara_points<ctype>(m_T, m_E);
        ptr_matsubara_quadrature_w = quadrature_provider.get_device_matsubara_weights<ctype>(m_T, m_E);
        reinit();
      }
    }

    /**
     * @brief Set the typical energy scale of the integrator and recompute the Matsubara quadrature rule.
     *
     * @param E The typical energy scale.
     */
    void set_E(const ctype E) { set_T(m_T, E); }

    IntegratorAngleFiniteTq0GPU(const IntegratorAngleFiniteTq0GPU &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes),
          device_data_size(other.device_data_size), ptr_x_quadrature_p(other.ptr_x_quadrature_p),
          ptr_x_quadrature_w(other.ptr_x_quadrature_w), ptr_ang_quadrature_p(other.ptr_ang_quadrature_p),
          ptr_ang_quadrature_w(other.ptr_ang_quadrature_w),
          ptr_matsubara_quadrature_p(other.ptr_matsubara_quadrature_p),
          ptr_matsubara_quadrature_w(other.ptr_matsubara_quadrature_w), x_extent(other.x_extent), m_T(other.m_T),
          pool(rmm::mr::get_current_device_resource(), (device_data_size / 256 + 1) * 256), m_E(other.m_E),
          manual_E(other.manual_E), max_block_size(other.max_block_size)
    {
      block_sizes = other.block_sizes;
      num_blocks = other.num_blocks;
      threads_per_block = other.threads_per_block;
    }

    template <typename... T> NT get(const ctype k, const T &...t)
    {
      if (!manual_E && (std::abs(k - m_E) / std::max(k, m_E) > 2.5e-2)) {
        set_T(m_T, k);
        manual_E = false;
      }

      const auto cuda_stream = cuda_stream_pool.get_stream();
      rmm::device_uvector<NT> device_data(device_data_size, cuda_stream, &pool);
      gridreduce_angle_finiteTq0<ctype, d, NT, KERNEL><<<num_blocks, threads_per_block, 0, cuda_stream.value()>>>(
          device_data.data(), ptr_x_quadrature_p, ptr_x_quadrature_w, ptr_ang_quadrature_p, ptr_ang_quadrature_w,
          ptr_matsubara_quadrature_p, ptr_matsubara_quadrature_w, x_extent, m_T, k, t...);
      check_cuda();
      return KERNEL::constant(k, t...) + thrust::reduce(thrust::cuda::par.on(cuda_stream.value()), device_data.begin(),
                                                        device_data.end(), NT(0.), thrust::plus<NT>());
    }

    template <typename... T> std::future<NT> request(const ctype k, const T &...t)
    {
      if (!manual_E && (std::abs(k - m_E) / std::max(k, m_E) > 2.5e-2)) {
        set_T(m_T, k);
        manual_E = false;
      }

      const auto cuda_stream = cuda_stream_pool.get_stream();
      std::shared_ptr<rmm::device_uvector<NT>> device_data =
          std::make_shared<rmm::device_uvector<NT>>(device_data_size, cuda_stream, &pool);
      gridreduce_angle_finiteTq0<ctype, d, NT, KERNEL><<<num_blocks, threads_per_block, 0, cuda_stream.value()>>>(
          (*device_data).data(), ptr_x_quadrature_p, ptr_x_quadrature_w, ptr_ang_quadrature_p, ptr_ang_quadrature_w,
          ptr_matsubara_quadrature_p, ptr_matsubara_quadrature_w, x_extent, m_T, k, t...);
      check_cuda();
      const NT constant = KERNEL::constant(k, t...);

      return std::async(std::launch::deferred, [=, this]() {
        return constant + thrust::reduce(thrust::cuda::par.on(cuda_stream.value()), (*device_data).begin(),
                                         (*device_data).end(), NT(0.), thrust::plus<NT>());
      });
    }

  private:
    QuadratureProvider &quadrature_provider;

    std::array<uint, 3> grid_sizes;
    std::array<uint, 3> block_sizes;

    uint device_data_size;

    const ctype *ptr_x_quadrature_p;
    const ctype *ptr_x_quadrature_w;
    const ctype *ptr_ang_quadrature_p;
    const ctype *ptr_ang_quadrature_w;
    const ctype *ptr_matsubara_quadrature_p;
    const ctype *ptr_matsubara_quadrature_w;

    const ctype x_extent;
    ctype m_T, m_E;
    bool manual_E;

    const uint max_block_size;
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
  template <int d, typename NT, typename KERNEL> class IntegratorAngleFiniteTq0GPU;
}

#else

#include <DiFfRG/physics/integration_finiteT/integrator_angle_finiteTq0_cpu.hh>

namespace DiFfRG
{
  template <int d, typename NT, typename KERNEL>
  class IntegratorAngleFiniteTq0GPU : public IntegratorAngleFiniteTq0TBB<d, NT, KERNEL>
  {
  public:
    using ctype = typename get_type::ctype<NT>;

    IntegratorAngleFiniteTq0GPU(QuadratureProvider &quadrature_provider, const std::array<uint, 3> _grid_sizes,
                                const ctype x_extent, const ctype T, const uint max_block_size = 256)
        : IntegratorAngleFiniteTq0TBB<d, NT, KERNEL>(quadrature_provider, _grid_sizes, x_extent, T, max_block_size)
    {
    }
  };
} // namespace DiFfRG

#endif

#endif