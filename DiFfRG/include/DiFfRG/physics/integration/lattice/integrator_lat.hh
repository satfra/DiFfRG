#pragma once

// DiFfRG
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/common/tbb.hh>
#include <DiFfRG/physics/integration/abstract_integrator.hh>
#include <DiFfRG/physics/integration/map_distribution.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>

// std
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief The momenta and weights of a d-dimensional lattice sum.
     *
     * Axis i runs over q_i = 2 pi n_i / (N_i a_i). An axis is summed either over the whole Brillouin
     * zone, n = -N/2 .. N/2 - 1 with weight 1, or -- for a kernel even in q_i -- over
     * n = 0 .. N/2 with weights 1, 2, ..., 2, 1, which is the same sum for half the cost. The
     * endpoint weights are what makes that exact: n = 0 and n = N/2 are their own mirror images.
     *
     * Axis 0 is the temporal one (N_t, a_t) and is halved only on request; all further axes are
     * spatial (N_s, a_s) and always halved, so a kernel must be even in each spatial component.
     */
    template <int d, typename ctype> struct LatticeGrid {
      /// Points visited per axis.
      device::array<size_t, d> count;
      /// n of the first point of each axis.
      device::array<int, d> n_first;
      /// N_i / 2, the mirror-symmetric endpoint of a halved axis.
      device::array<int, d> n_half;
      device::array<bool, d> halved;
      /// 2 pi / (N_i a_i)
      device::array<ctype, d> dq;
      /// 1 / prod_i (N_i a_i), the measure of one lattice point.
      ctype measure;

      LatticeGrid() = default;
      LatticeGrid(const device::array<uint, d> &N, const device::array<ctype, d> &a, const bool q0_symmetric)
      {
        measure = 1;
        for (int i = 0; i < d; ++i) {
          if (N[i] < 2 || N[i] % 2 != 0)
            throw std::runtime_error("IntegratorLat: lattice sizes must be even and at least 2, got " +
                                     std::to_string(N[i]));
          halved[i] = i > 0 || q0_symmetric;
          n_half[i] = int(N[i] / 2);
          n_first[i] = halved[i] ? 0 : -n_half[i];
          count[i] = halved[i] ? N[i] / 2 + 1 : N[i];
          dq[i] = ctype(2 * M_PI) / (ctype(N[i]) * a[i]);
          measure /= ctype(N[i]) * a[i];
        }
      }

      size_t size() const
      {
        size_t s = 1;
        for (int i = 0; i < d; ++i)
          s *= count[i];
        return s;
      }

      /// Momentum and weight (including the measure) of the point with multi-index idx.
      KOKKOS_INLINE_FUNCTION ctype point(const device::array<size_t, d> &idx, device::array<ctype, d> &q) const
      {
        ctype weight = measure;
        for (int i = 0; i < d; ++i) {
          const int n = n_first[i] + int(idx[i]);
          q[i] = dq[i] * ctype(n);
          if (halved[i] && n != 0 && n != n_half[i]) weight *= 2;
        }
        return weight;
      }

      /// Row-major: the last axis runs fastest.
      KOKKOS_INLINE_FUNCTION device::array<size_t, d> unflatten(size_t flat) const
      {
        device::array<size_t, d> idx;
        for (int i = d - 1; i >= 0; --i) {
          idx[i] = flat % count[i];
          flat /= count[i];
        }
        return idx;
      }
    };
  } // namespace internal

  /**
   * @brief Sums a kernel over the momenta of a d-dimensional periodic lattice, d = 1..4.
   *
   *     result = constant(args...) + 1/prod_i(N_i a_i) sum_{n} kernel(q_0, ..., q_{d-1}, args...)
   *
   * with q_i = 2 pi n_i / (N_i a_i) and n_i in the first Brillouin zone. Axis 0 has extent N_t and
   * spacing a_t, every further axis N_s and a_s; see internal::LatticeGrid for which axes are halved
   * and what that assumes about the kernel.
   *
   * map() is distributed over MPI ranks by MapScheduler, exactly like the quadrature integrators.
   *
   * @tparam d number of lattice dimensions
   * @tparam NT numerical type of the result
   * @tparam KERNEL provides static `kernel(q_0, ..., q_{d-1}, args...)` and `constant(args...)`
   * @tparam ExecutionSpace GPU_exec, KokkosHost_exec or TBB_exec
   */
  template <int d, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(d >= 1 && d <= 4)
  class IntegratorLat : public AbstractIntegrator
  {
  public:
    using ctype = typename get_type::ctype<NT>;
    using execution_space = ExecutionSpace;

    /// (N_t) and (a_t) for d == 1, (N_t, N_s) and (a_t, a_s) otherwise.
    static constexpr size_t n_extents = d == 1 ? 1 : 2;

    IntegratorLat(const std::array<uint, n_extents> grid_size, const std::array<ctype, n_extents> a,
                  const bool q0_symmetric = false)
        : IntegratorLat(ExecutionSpace(), grid_size, a, q0_symmetric)
    {
    }

    /**
     * @brief Construct from /integration/lattice/{N_t, N_s, a_t, a_s, q0_symmetric}, as generated flows do.
     *
     * d == 1 reads only N_t and a_t. q0_symmetric is optional and defaults to false.
     */
    IntegratorLat(QuadratureProvider &quadrature_provider, const ConfigTree &config)
        : IntegratorLat(quadrature_provider.template next_execution_space<ExecutionSpace>(), config_sizes(config),
                        config_spacings(config), config.get_bool("/integration/lattice/q0_symmetric", false))
    {
    }

    void set_a(const std::array<ctype, n_extents> a)
    {
      m_a = a;
      m_grid = make_grid();
    }
    void set_q0_symmetric(const bool symmetric)
    {
      m_q0_symmetric = symmetric;
      m_grid = make_grid();
    }

    const std::array<uint, n_extents> &grid_size() const { return m_grid_size; }

    /// Lattice points visited per external grid point. The scheduler's cost score.
    size_t quadrature_volume() const { return m_grid.size(); }

    template <typename... T>
      requires internal::accepts_args<NT, KERNEL, ctype, d, T...>
    void get(NT &dest, const T &...t) const
    {
      if (!m_result_views_initialized) {
        m_result_view = Kokkos::View<NT, typename ExecutionSpace::memory_space>("IntegratorLat_result");
        m_result_host = Kokkos::create_mirror_view(m_result_view);
        m_result_views_initialized = true;
      }
      reduce(m_space, m_result_view, t...);
      Kokkos::deep_copy(m_space, m_result_host, m_result_view);
      m_space.fence();
      dest = m_result_host();
    }

    /// Single-precision summation handing back a double result.
    template <typename OT, typename... T>
      requires(internal::is_widened_result<OT, NT> && internal::accepts_args<NT, KERNEL, ctype, d, T...>)
    void get(OT &dest, const T &...t) const
    {
      NT result;
      get(result, t...);
      dest = OT(result);
    }

    /// The reduction behind get(). Public only because nvcc rejects extended device lambdas inside
    /// non-public member functions.
    template <typename ResultView, typename... T>
    void reduce(const ExecutionSpace &space, const ResultView &result, const T &...t) const
    {
      constexpr bool handles = internal::takes_handles<NT, KERNEL, ctype, d, T...>;
      internal::check_kernel_precision<handles, ctype, T...>();
      const auto args = device::make_tuple(internal::to_launch_arg<handles, ctype>(t)...);
      const auto grid = m_grid;

      Kokkos::parallel_reduce(
          "IntegratorLat_" + std::to_string(d) + "D", Kokkos::RangePolicy<ExecutionSpace>(space, 0, grid.size()),
          KOKKOS_LAMBDA(const size_t flat, NT &update) {
            device::array<ctype, d> q;
            const ctype weight = grid.point(grid.unflatten(flat), q);
            device::apply(
                [&](const auto &...qs) {
                  device::apply([&](const auto &...as) { update += weight * KERNEL::kernel(qs..., as...); }, args);
                },
                q);
            // The constant goes in on the device: a kernel argument may be a device-side handle.
            if (flat == 0) device::apply([&](const auto &...as) { update += KERNEL::constant(as...); }, args);
          },
          result);
    }

    /**
     * @brief Sum at every point of `coordinates` into the device view `integral_view`.
     *
     * One team per point; the kernel receives the lattice momenta, then the point's position, then
     * `args`.
     */
    template <typename view_type, typename Coordinates, typename... Args>
    void map(ExecutionSpace &space, const view_type integral_view, const Coordinates &coordinates, const Args &...args)
    {
      using OT = typename view_type::value_type;
      using TeamType = typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;

      constexpr bool handles = internal::map_takes_handles<NT, KERNEL, ctype, d, Coordinates, Args...>;
      internal::check_kernel_precision<handles, ctype, Args...>();
      const auto m_args = device::make_tuple(internal::to_kernel_arg<handles, ctype>(args)...);
      const auto grid = m_grid;
      const size_t total = grid.size();

      Kokkos::parallel_for(
          "IntegratorLat_map_" + std::to_string(d) + "D",
          Kokkos::TeamPolicy<ExecutionSpace>(space, integral_view.size(), Kokkos::AUTO),
          KOKKOS_LAMBDA(const TeamType &team) {
            const size_t k = team.league_rank();
            const auto pos = coordinates.forward(coordinates.from_linear_index(k));

            // Nested packs rather than one tuple_cat: m_args holds every interpolator by value, and
            // a concatenated tuple would be a per-thread copy of all of them.
            NT res{};
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, total),
                [&](const size_t flat, NT &update) {
                  device::array<ctype, d> q;
                  const ctype weight = grid.point(grid.unflatten(flat), q);
                  device::apply(
                      [&](const auto &...qs) {
                        device::apply(
                            [&](const auto &...ps) {
                              device::apply(
                                  [&](const auto &...as) { update += weight * KERNEL::kernel(qs..., ps..., as...); },
                                  m_args);
                            },
                            pos);
                      },
                      q);
                },
                res);

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              device::apply(
                  [&](const auto &...ps) {
                    device::apply(
                        [&](const auto &...as) {
                          integral_view(k) = static_cast<OT>(res + KERNEL::constant(ps..., as...));
                        },
                        m_args);
                  },
                  pos);
            });
          });
    }

    template <typename OT, typename Coordinates, typename... Args>
      requires internal::is_map_result<OT, NT>
    auto map(OT *dest, const Coordinates &coordinates, const Args &...args)
    {
      internal::scheduled_map<ExecutionSpace>(integrator_id(), quadrature_volume(), dest, coordinates,
                                              [&](auto *d_, const auto &c) { this->map_dist(d_, c, args...); });
      return m_space;
    }

    /// map() without the MapScheduler: computes the whole of `coordinates` on this rank.
    template <typename OT, typename Coordinates, typename... Args>
      requires internal::is_map_result<OT, NT>
    auto map_dist(OT *dest, const Coordinates &coordinates, const Args &...args)
    {
      internal::staged_map(m_space, m_staging.template get<OT>(), dest, coordinates.size(),
                           [this, coordinates, args...](const auto &view) {
                             this->map(this->m_space, view, coordinates, args...);
                           });
      return m_space;
    }

  protected:
    IntegratorLat(const ExecutionSpace &space, const std::array<uint, n_extents> grid_size,
                  const std::array<ctype, n_extents> a, const bool q0_symmetric)
        : m_grid_size(grid_size), m_a(a), m_q0_symmetric(q0_symmetric), m_grid(make_grid()), m_space(space)
    {
    }

    internal::LatticeGrid<d, ctype> make_grid() const
    {
      device::array<uint, d> N;
      device::array<ctype, d> a;
      // Axis 0 takes the temporal extent, every further axis the spatial one.
      for (int i = 0; i < d; ++i) {
        N[i] = m_grid_size[std::min<size_t>(i, n_extents - 1)];
        a[i] = m_a[std::min<size_t>(i, n_extents - 1)];
      }
      return internal::LatticeGrid<d, ctype>(N, a, m_q0_symmetric);
    }

    static std::array<uint, n_extents> config_sizes(const ConfigTree &config)
    {
      if constexpr (d == 1)
        return {config.get_uint("/integration/lattice/N_t")};
      else
        return {config.get_uint("/integration/lattice/N_t"), config.get_uint("/integration/lattice/N_s")};
    }

    static std::array<ctype, n_extents> config_spacings(const ConfigTree &config)
    {
      if constexpr (d == 1)
        return {ctype(config.get_double("/integration/lattice/a_t"))};
      else
        return {ctype(config.get_double("/integration/lattice/a_t")),
                ctype(config.get_double("/integration/lattice/a_s"))};
    }

    std::array<uint, n_extents> m_grid_size;
    std::array<ctype, n_extents> m_a;
    bool m_q0_symmetric;
    internal::LatticeGrid<d, ctype> m_grid;

    /// Mutable because the const get() issues work on it: which stream a launch goes to is not part
    /// of the integrator's logical state.
    mutable ExecutionSpace m_space;
    mutable internal::MapStagingSet<NT, ExecutionSpace> m_staging;
    mutable Kokkos::View<NT, typename ExecutionSpace::memory_space> m_result_view;
    mutable typename Kokkos::View<NT, typename ExecutionSpace::memory_space>::host_mirror_type m_result_host;
    mutable bool m_result_views_initialized = false;
  };

  /**
   * @brief The TBB variant: reductions through TBBReduction, so the result does not depend on the
   * thread count, and map() as one get() per point.
   */
  template <int d, typename NT, typename KERNEL>
    requires(d >= 1 && d <= 4)
  class IntegratorLat<d, NT, KERNEL, TBB_exec> : public IntegratorLat<d, NT, KERNEL, KokkosHost_exec>
  {
    using Base = IntegratorLat<d, NT, KERNEL, KokkosHost_exec>;

  public:
    using ctype = typename get_type::ctype<NT>;
    using execution_space = TBB_exec;

    IntegratorLat(const std::array<uint, Base::n_extents> grid_size, const std::array<ctype, Base::n_extents> a,
                  const bool q0_symmetric = false)
        : Base(grid_size, a, q0_symmetric)
    {
    }

    IntegratorLat(QuadratureProvider &quadrature_provider, const ConfigTree &config)
        : Base(quadrature_provider, config)
    {
    }

    template <typename... T>
      requires internal::accepts_args<NT, KERNEL, ctype, d, T...>
    void get(NT &dest, const T &...t) const
    {
      constexpr bool handles = internal::takes_handles<NT, KERNEL, ctype, d, T...>;
      internal::check_kernel_precision<handles, ctype, T...>();
      if constexpr (!(std::is_same_v<T, internal::launch_arg_t<handles, T, ctype>> && ...))
        get(dest, internal::to_launch_arg<handles, ctype>(t)...);
      else {
        const auto args = device::tie(t...);
        const auto &grid = this->m_grid;
        auto functor = [&](const device::array<size_t, d> &idx) {
          device::array<ctype, d> q;
          const ctype weight = grid.point(idx, q);
          return device::apply([&](const auto &...iargs) { return weight * KERNEL::kernel(iargs...); },
                               device::tuple_cat(q, args));
        };
        dest = KERNEL::constant(t...) + TBBReduction<d, NT, decltype(functor)>(grid.count, functor);
      }
    }

    template <typename OT, typename... T>
      requires(internal::is_widened_result<OT, NT> && internal::accepts_args<NT, KERNEL, ctype, d, T...>)
    void get(OT &dest, const T &...t) const
    {
      NT result;
      get(result, t...);
      dest = OT(result);
    }

    template <typename OT, typename Coordinates, typename... Args>
      requires internal::is_map_result<OT, NT>
    void map(execution_space &, OT *dest, const Coordinates &coordinates, const Args &...args)
    {
      // Handles once here rather than per point in get().
      constexpr bool handles = internal::map_takes_handles<NT, KERNEL, ctype, d, Coordinates, Args...>;
      internal::check_kernel_precision<handles, ctype, Args...>();
      const auto m_args = [&] {
        if constexpr (handles)
          return device::make_tuple(DiFfRG::to_kernel_handle<ctype>(args)...);
        else
          return device::tie(args...);
      }();

      tbb::parallel_for(tbb::blocked_range<size_t>(0, coordinates.size()), [&](const tbb::blocked_range<size_t> &r) {
        for (size_t idx = r.begin(); idx != r.end(); ++idx) {
          const auto pos = coordinates.forward(coordinates.from_linear_index(idx));
          device::apply(
              [&](const auto &...pargs) {
                device::apply([&](const auto &...iargs) { get(dest[idx], pargs..., iargs...); }, m_args);
              },
              pos);
        }
      });
    }

    template <typename OT, typename Coordinates, typename... Args>
      requires internal::is_map_result<OT, NT>
    auto map(OT *dest, const Coordinates &coordinates, const Args &...args)
    {
      internal::scheduled_map<execution_space>(
          this->integrator_id(), Base::quadrature_volume(), dest, coordinates, [&](auto *d_, const auto &c) {
            // tbb::parallel_for writes straight into `dest`, so there is nothing to stage.
            internal::run_or_queue_host([this, d_, c, args...]() {
              auto sp = execution_space();
              this->map(sp, d_, c, args...);
            });
            if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
          });
      return execution_space();
    }

    /// map() without the MapScheduler: computes the whole of `coordinates` on this rank.
    template <typename OT, typename Coordinates, typename... Args>
      requires internal::is_map_result<OT, NT>
    auto map_dist(OT *dest, const Coordinates &coordinates, const Args &...args)
    {
      auto sp = execution_space();
      map(sp, dest, coordinates, args...);
      return sp;
    }
  };

  template <typename NT, typename KERNEL, typename ExecutionSpace>
  using IntegratorLat1D = IntegratorLat<1, NT, KERNEL, ExecutionSpace>;
  template <typename NT, typename KERNEL, typename ExecutionSpace>
  using IntegratorLat2D = IntegratorLat<2, NT, KERNEL, ExecutionSpace>;
  template <typename NT, typename KERNEL, typename ExecutionSpace>
  using IntegratorLat3D = IntegratorLat<3, NT, KERNEL, ExecutionSpace>;
  template <typename NT, typename KERNEL, typename ExecutionSpace>
  using IntegratorLat4D = IntegratorLat<4, NT, KERNEL, ExecutionSpace>;
} // namespace DiFfRG
