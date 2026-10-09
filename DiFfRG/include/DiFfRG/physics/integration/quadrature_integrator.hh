#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/common/tbb.hh>
#include <DiFfRG/common/tuples.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/abstract_integrator.hh>
#include <DiFfRG/physics/integration/map_completion.hh>
#include <DiFfRG/physics/integration/map_distribution.hh>
#include <DiFfRG/physics/integration/point_arg.hh>
#include <DiFfRG/physics/interpolation/interpolator_handle.hh>

// std
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

namespace DiFfRG
{
  /**
   * @brief Whether a coordinates type carries enough identity for QuadratureIntegrator::map() to
   * cache its forward()-transformed positions in a device view (one forward() per grid point
   * instead of per thread). Namespace-scope on purpose: it is referenced inside extended device
   * lambdas, where nvcc mishandles function-local constexpr variables.
   */
  template <typename Coordinates>
  inline constexpr bool has_cacheable_positions_v = requires(const Coordinates &c) {
    c.to_string();
    c.forward(c.from_linear_index(size_t(0)));
  };

  namespace internal
  {
    /// A single-precision integrator may hand its results back in double precision.
    template <typename OT, typename NT>
    inline constexpr bool is_widened_result =
        !std::is_same_v<OT, NT> && std::is_same_v<OT, get_type::double_precision<NT>>;

    /// Element types an integrator with value type NT can map() into.
    template <typename OT, typename NT>
    inline constexpr bool is_map_result = std::is_same_v<OT, NT> || is_widened_result<OT, NT>;

    /// What a kernel computing in ctype receives for an argument of type T: an interpolator's
    /// handle (see has_kernel_handle), anything else in the kernel's precision.
    template <typename T, typename ctype> using kernel_arg_t = compute_arg_t<kernel_handle_t<T, ctype>, ctype>;

    /// Whether KERNEL takes handles. Kernels generated before handles existed name the interpolator
    /// types in their signature; they are passed the interpolators themselves.
    template <typename NT, typename KERNEL, typename ctype, int dim, typename... T>
    inline constexpr bool takes_handles = is_valid_kernel<NT, KERNEL, ctype, dim, kernel_arg_t<T, ctype>...>;

    template <size_t, typename T> using repeat_t = T;

    /// takes_handles for map(), whose kernel also receives the cdim grid positions of Coordinates.
    template <typename NT, typename KERNEL, typename ctype, int dim, typename Coordinates, typename... Args>
    inline constexpr bool map_takes_handles = []<size_t... I>(std::index_sequence<I...>) {
      return takes_handles<NT, KERNEL, ctype, dim, repeat_t<I, typename Coordinates::ctype>..., Args...>;
    }(std::make_index_sequence<Coordinates::dim>{});

    template <typename NT, typename KERNEL, typename ctype, int dim, typename... T>
    concept accepts_args = takes_handles<NT, KERNEL, ctype, dim, T...> ||
                           is_valid_kernel<NT, KERNEL, ctype, dim, compute_arg_t<T, ctype>...>;

    template <bool handles, typename T, typename ctype>
    using launch_arg_t = std::conditional_t<handles, kernel_arg_t<T, ctype>, compute_arg_t<T, ctype>>;

    /// An argument of get() or map_points() as the kernel receives it.
    template <bool handles, typename ctype, typename T> launch_arg_t<handles, T, ctype> to_launch_arg(const T &t)
    {
      if constexpr (handles)
        return launch_arg_t<handles, T, ctype>(to_kernel_handle<ctype>(t));
      else
        return launch_arg_t<handles, T, ctype>(t);
    }

    /// An argument as the kernel receives it. Unlike to_launch_arg, scalars keep their precision.
    template <bool handles, typename ctype, typename T> decltype(auto) to_kernel_arg(const T &t)
    {
      if constexpr (handles)
        return to_kernel_handle<ctype>(t);
      else
        return (t);
    }

    // Through if constexpr: in a plain `has_kernel_handle<T> && ...` the right operand still names
    // T::value_type, a hard error for argument types such as double that have none.
    template <typename T> constexpr bool is_double_interpolator_impl()
    {
      if constexpr (has_kernel_handle<T>)
        return std::is_same_v<typename T::value_type, double> ||
               std::is_same_v<typename T::value_type, complex<double>>;
      else
        return false;
    }
    template <typename T> inline constexpr bool is_double_interpolator = is_double_interpolator_impl<T>();

    [[deprecated("a single-precision kernel names double interpolator types, so it is passed the interpolators "
                 "and looks them up in double. Regenerate it: a kernel taking its interpolators as const auto& "
                 "reads their single-precision copy.")]]
    constexpr void double_lookups_in_float_kernel()
    {
    }

    /// Warns at compile time when a float kernel would silently read double interpolators.
    template <bool handles, typename ctype, typename... T> constexpr void check_kernel_precision()
    {
      if constexpr (std::is_same_v<ctype, float> && !handles && (is_double_interpolator<T> || ...))
        double_lookups_in_float_kernel();
    }

    /// f(begin, end) over [0, n), in parallel chunks once n is large enough to pay for it.
    template <typename F> void parallel_chunks(const size_t n, const F &f)
    {
      constexpr size_t grain = 1 << 15;
      if (n < 2 * grain)
        f(size_t(0), n);
      else
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, grain),
                          [&](const tbb::blocked_range<size_t> &r) { f(r.begin(), r.end()); });
    }

    /// Grow-only byte buffer, used to stage the per-point arguments of map_points().
    template <typename MemorySpace> struct ByteBuffer {
      Kokkos::View<char *, MemorySpace> view;

      char *reserve(const size_t bytes)
      {
        if (view.extent(0) < bytes)
          view = Kokkos::View<char *, MemorySpace>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "PointArgs"), bytes);
        return view.data();
      }
    };

    /// Types that cannot be placed in map_points' staging buffer (an interpolator: no default constructor, no copy
    /// assignment) can only be shared by all points, and are passed to the device by value. A variable template, not
    /// a constexpr generic lambda: CUDA 12's front end aborts on calling the latter's operator()<U>().
    template <typename U>
    inline constexpr bool is_stageable = std::is_default_constructible_v<U> && std::is_copy_assignable_v<U>;

    /**
     * @brief The buffers of a device map_points(): its per-point arguments, staged on the host and copied to the
     * device in one go, and its results. Kept by the integrator across calls.
     */
    template <typename NT, typename ExecutionSpace> struct MapPointsBuffers {
      using memory_space = typename ExecutionSpace::memory_space;
      static constexpr bool host_memory = std::is_same_v<memory_space, CPU_memory>;

      ByteBuffer<PinnedHost_memory> args_host;
      ByteBuffer<memory_space> args_device;
      MapStagingSet<NT, ExecutionSpace> result;

      /**
       * @brief Copy the per-point arguments to the device, converted to the compute precision ctype, and return
       * them as a tuple of DevicePointArg.
       *
       * All per-point arrays share one staging buffer, so the upload is a single copy. On a host memory space an
       * argument that needs no conversion is used in place.
       */
      template <typename ctype, bool handles, typename... T>
      auto stage(const ExecutionSpace &space, const size_t n, const PointArg<T> &...args)
      {
        constexpr size_t alignment = 64;
        constexpr size_t n_args = sizeof...(T);

        // Byte offset of each staged argument; arguments that are broadcast, or used in place, stay unstaged.
        std::array<size_t, n_args> offsets{};
        std::array<bool, n_args> staged{};
        size_t total = 0;
        {
          size_t k = 0;
          (
              [&] {
                using U = launch_arg_t<handles, T, ctype>;
                args.check_size(n);
                if constexpr (!is_stageable<U>)
                  if (args.per_point())
                    throw std::runtime_error("map_points: this argument type can only be shared by all points.");
                staged[k] = args.per_point() && !(host_memory && std::is_same_v<U, T>);
                if (staged[k]) {
                  offsets[k] = total;
                  total += (n * sizeof(U) + alignment - 1) / alignment * alignment;
                }
                ++k;
              }(),
              ...);
        }

        char *host = nullptr;
        char *device = nullptr;
        if (total > 0) {
          host = args_host.reserve(total);
          device = host_memory ? host : args_device.reserve(total);
          size_t k = 0;
          (
              [&] {
                using U = launch_arg_t<handles, T, ctype>;
                if constexpr (is_stageable<U>)
                  if (staged[k]) {
                    U *out = reinterpret_cast<U *>(host + offsets[k]);
                    parallel_chunks(n, [&](const size_t begin, const size_t end) {
                      for (size_t i = begin; i < end; ++i)
                        out[i] = to_launch_arg<handles, ctype>(args.values[i]);
                    });
                  }
                ++k;
              }(),
              ...);
          if constexpr (!host_memory)
            Kokkos::deep_copy(space, Kokkos::subview(args_device.view, Kokkos::make_pair(size_t(0), total)),
                              Kokkos::subview(args_host.view, Kokkos::make_pair(size_t(0), total)));
        }

        return device_args<ctype, handles>(std::index_sequence_for<T...>{}, staged, offsets, device, args...);
      }

      // The tuple of DevicePointArg that stage() returns. Plain function templates rather than a lambda expanded
      // over two packs inside a templated lambda: CUDA 12's device front end (cicc) aborts on the latter.
      template <typename ctype, bool handles, size_t... I, typename... T>
      static auto device_args(std::index_sequence<I...>, const std::array<bool, sizeof...(T)> &staged,
                              const std::array<size_t, sizeof...(T)> &offsets, const char *device,
                              const PointArg<T> &...args)
      {
        return device::make_tuple(device_arg<ctype, handles>(staged[I], offsets[I], device, args)...);
      }

      template <typename ctype, bool handles, typename T>
      static auto device_arg(const bool staged, const size_t offset, const char *device, const PointArg<T> &arg)
      {
        using U = launch_arg_t<handles, T, ctype>;
        if constexpr (!is_stageable<U>)
          return DevicePointArg<U>{nullptr, to_launch_arg<handles, ctype>(arg.value)};
        else {
          const U *values = staged ? reinterpret_cast<const U *>(device + offset)
                                   : (arg.per_point() ? reinterpret_cast<const U *>(arg.values) : nullptr);
          return DevicePointArg<U>{values, arg.per_point() ? U{} : to_launch_arg<handles, ctype>(arg.value)};
        }
      }

      /**
       * @brief Run launch(result_view) and copy its n results to @p dest: in place on a host memory space,
       * otherwise through a device buffer and page-locked staging.
       */
      template <typename OT, typename Launch>
      void run(ExecutionSpace &space, const PointSpan<OT> dest, const Launch &launch)
      {
        const size_t n = dest.size();
        if constexpr (host_memory) {
          launch(Kokkos::View<OT *, CPU_memory, Kokkos::MemoryUnmanaged>(dest.data(), n));
          space.fence();
        } else {
          auto &stage = result.template get<OT>();
          const auto device_result = stage.device_view(space, n);
          const auto pinned = stage.pinned_view(n);
          launch(device_result);
          Kokkos::deep_copy(space, pinned, device_result);
          space.fence();
          parallel_chunks(n, [&](const size_t begin, const size_t end) {
            std::memcpy(dest.data() + begin, pinned.data() + begin, (end - begin) * sizeof(OT));
          });
        }
      }
    };

    /// map_points() of a host integrator: integrator.get() at every point, in a flat parallel loop.
    template <typename Integrator, typename OT, typename... T>
    void map_points_by_get(const Integrator &integrator, const PointSpan<OT> dest, const PointArg<T> &...args)
    {
      (args.check_size(dest.size()), ...);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, dest.size()), [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i != r.end(); ++i)
          integrator.get(dest[i], args[i]...);
      });
    }
  } // namespace internal

  /**
   * @brief How a device map_points() distributes its work.
   *
   * thread_per_point: one thread sums all quadrature nodes of one point; only competitive for very
   * many points with few nodes. team_per_point: one team reduces over the nodes of one point; the
   * better choice almost everywhere. automatic picks between the two from these sizes.
   */
  enum class MapPointsPolicy { automatic, thread_per_point, team_per_point };

  namespace internal
  {
    /// The policy a device map_points() over n points of total quadrature nodes each runs with. Measured on an
    /// RTX 4070 (ONfiniteT, 64..2e5 points, 16..8192 nodes): a team per point wins nearly everywhere, by up to
    /// 18x; a thread per point only pays off for very many cheap points.
    inline MapPointsPolicy resolve_policy(const MapPointsPolicy policy, const size_t n, const size_t total)
    {
      if (policy != MapPointsPolicy::automatic) return policy;
      return (n >= 65536 && total <= 1024) ? MapPointsPolicy::thread_per_point : MapPointsPolicy::team_per_point;
    }
  } // namespace internal

  /**
   * @brief This class performs numerical integration over a d-dimensional hypercube using quadrature rules.
   *
   * @tparam dim The dimension of the hypercube, which can be between 1 and 5.
   * @tparam NT numerical type of the result
   * @tparam KERNEL kernel to be integrated, which must provide the static methods `kernel` and `constant`
   * @tparam ExecutionSpace can be any execution space, e.g. GPU_exec, TBB_exec.
   */
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(dim > 0)
  class QuadratureIntegrator : public AbstractIntegrator
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    /**
     * @brief Execution space to be used for the integration, e.g. GPU_exec, TBB_exec.
     */
    using execution_space = ExecutionSpace;

    QuadratureIntegrator(QuadratureProvider &quadrature_provider, const std::array<size_t, dim> &_grid_size,
                         const std::array<ctype, dim> &grid_min, const std::array<ctype, dim> &grid_max,
                         const std::array<QuadratureType, dim> &quadrature_type)
        : space(quadrature_provider.template next_execution_space<ExecutionSpace>()),
          quadrature_provider(quadrature_provider)
    {
      for (size_t i = 0; i < dim; ++i) {
        grid_size[i] = _grid_size[i];

        nodes[i] = quadrature_provider.template nodes<ctype, typename ExecutionSpace::memory_space>(grid_size[i],
                                                                                                    quadrature_type[i]);
        weights[i] = quadrature_provider.template weights<ctype, typename ExecutionSpace::memory_space>(
            grid_size[i], quadrature_type[i]);
      }
      set_grid_extents(grid_min, grid_max);
    }

    void set_grid_extents(const std::array<ctype, dim> &grid_min, const std::array<ctype, dim> &grid_max)
    {
      for (size_t i = 0; i < dim; ++i) {
        grid_extents[0][i] = grid_min[i];
        grid_extents[1][i] = grid_max[i];

        grid_start[i] = grid_extents[0][i];
        grid_scale[i] = (grid_extents[1][i] - grid_extents[0][i]);
      }
    }

    template <typename... T>
      requires internal::accepts_args<NT, KERNEL, ctype, dim, T...>
    void get(NT &dest, const T &...t) const
    {
      // create an execution space
      ExecutionSpace space;

      if (!m_result_views_initialized) {
        m_result_view = Kokkos::View<NT, typename ExecutionSpace::memory_space>("result");
        m_result_host = Kokkos::create_mirror_view(m_result_view);
        m_result_views_initialized = true;
      }
      get(space, m_result_view, t...);
      Kokkos::deep_copy(space, m_result_host, m_result_view);
      space.fence();
      dest = m_result_host();
    }

    /// Single-precision integration handing back a double result.
    template <typename OT, typename... T>
      requires(internal::is_widened_result<OT, NT> &&
               internal::accepts_args<NT, KERNEL, ctype, dim, T...>)
    void get(OT &dest, const T &...t) const
    {
      NT result;
      get(result, t...);
      dest = OT(result);
    }

    template <typename OT, typename... T>
      requires(!std::is_same_v<OT, NT> && !internal::is_widened_result<OT, NT> &&
               internal::accepts_args<NT, KERNEL, ctype, dim, T...>)
    void get(OT &dest, const T &...t) const
    {
      ExecutionSpace space;
      get(space, dest, t...);
    }

    template <typename OT, typename... T>
      requires(!std::is_same_v<OT, NT> && !internal::is_widened_result<OT, NT> &&
               internal::accepts_args<NT, KERNEL, ctype, dim, T...>)
    void get(ExecutionSpace &space, OT &dest, const T &...t) const
    {
      constexpr bool handles = internal::takes_handles<NT, KERNEL, ctype, dim, T...>;
      internal::check_kernel_precision<handles, ctype, T...>();
      const auto args = device::make_tuple(internal::to_launch_arg<handles, ctype>(t)...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      auto functor = KOKKOS_LAMBDA(const device::array<size_t, dim> &idx, NT &update)
      {
        device::array<ctype, dim> x;
        ctype weight = 1;
        bool is_first = true;
        for (size_t i = 0; i < dim; ++i) {
          x[i] = Kokkos::fma(scale[i], n[i][idx[i]], start[i]);
          weight *= w[i][idx[i]] * scale[i];
          is_first &= idx[i] == 0;
        }
        device::apply([&](const auto &...iargs) { update += weight * KERNEL::kernel(iargs...); },
                      device::tuple_cat(x, args));
        device::apply([&](const auto &...iargs) { update += is_first ? KERNEL::constant(iargs...) : NT(0); }, args);
      };

      Kokkos::parallel_reduce("QuadratureIntegral_" + std::to_string(dim) + "D", // name of the kernel
                              make_kokkos_nd_range<dim, ExecutionSpace>(space, {0}, grid_size),
                              KokkosNDLambdaWrapperReduction<dim, decltype(functor)>(functor), dest);
    }

    template <typename view_type, typename Coordinates, typename... Args>
    void map(ExecutionSpace &space, const view_type integral_view, const Coordinates &coordinates, const Args &...args)
    {
      device::array<size_t, 1 + dim> extents;
      extents[0] = integral_view.size();
      for (int i = 0; i < dim; ++i)
        extents[1 + i] = grid_size[i];

      // Reuse cached view if large enough, otherwise reallocate (grow-only)
      {
        bool needs_realloc = false;
        for (size_t i = 0; i < 1 + dim; ++i)
          needs_realloc |= (extents[i] > m_cache_extents[i]);
        if (needs_realloc) {
          for (size_t i = 0; i < 1 + dim; ++i)
            m_cache_extents[i] = std::max(m_cache_extents[i], extents[i]);
          m_cache = make_kokkos_nd_view<1 + dim, NT, ExecutionSpace>("cache", m_cache_extents);
        }
      }
      // Create a Restrict-tagged alias of the cache for no-alias optimization
      const auto cache = KokkosNDViewRestrict<1 + dim, NT, ExecutionSpace>(m_cache);

      constexpr bool handles = internal::map_takes_handles<NT, KERNEL, ctype, dim, Coordinates, Args...>;
      internal::check_kernel_precision<handles, ctype, Args...>();
      const auto m_args = device::make_tuple(internal::to_kernel_arg<handles, ctype>(args)...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      // The external position is a function of idx[0] alone: at most integral_view.size() distinct
      // values per launch, while the functor below runs size * prod(grid_size) threads. For the
      // logarithmic coordinate classes forward() is a fp64 expm1/sinh+exp, so recomputing it per
      // thread wastes one transcendental per thread. Precompute the positions once per coordinate
      // system into a device view; the fill runs the very same forward() on the same device, so the
      // cached values are bit-identical to what the per-thread computation produced.
      //
      // Coordinates without a to_string() identity keep the per-thread computation. SubCoordinates
      // (the MPI path) is *not* one of them -- it has its own to_string(), and the key construction
      // below was written to separate two windows into the same base grid. The only cost on that
      // path is that a schedule which moves a slice boundary re-keys and re-runs the ~5 us fill.
      constexpr size_t cdim = Coordinates::dim;
      // Compile-time, so the un-taken branch in the device functors below is dead code and the
      // per-thread forward() (a fp64 expm1/sinh+exp on the log grids) actually leaves the kernel.
      if constexpr (has_cacheable_positions_v<Coordinates>) {
        // The key must identify the positions, not just the coordinate parameters. SubCoordinates
        // does put its window into its own to_string(), but appending the exact (hex-formatted)
        // first and last positions makes the separation independent of that.
        std::string key = coordinates.to_string() + "|" + std::to_string(integral_view.size());
        {
          char buf[64];
          const auto first = coordinates.forward(coordinates.from_linear_index(size_t(0)));
          const auto last = coordinates.forward(coordinates.from_linear_index(integral_view.size() - 1));
          for (size_t d = 0; d < cdim; ++d) {
            std::snprintf(buf, sizeof(buf), "|%la|%la", double(first[d]), double(last[d]));
            key += buf;
          }
        }
        const size_t need = integral_view.size() * cdim;
        if (m_positions_key != key || m_positions.extent(0) < need) {
          if (m_positions.extent(0) < need)
            m_positions = Kokkos::View<ctype *, typename ExecutionSpace::memory_space>(
                Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "QuadratureIntegrator_positions"), need);
          const auto pos_fill = m_positions;
          const auto coords = coordinates;
          Kokkos::parallel_for(
              "QuadratureIntegrator_fill_positions",
              Kokkos::RangePolicy<ExecutionSpace>(space, 0, integral_view.size()), KOKKOS_LAMBDA(const size_t i) {
                const auto p = coords.forward(coords.from_linear_index(i));
                for (size_t d = 0; d < cdim; ++d)
                  pos_fill(i * cdim + d) = p[d];
              });
          m_positions_key = key;
        }
      }
      const auto pos_view = m_positions;
      using pos_ctype = typename Coordinates::ctype;
      // Runtime copy for the team lambda below: its constant() evaluation runs once per team, so
      // keeping the transform code there costs nothing, and a plain branch avoids nvcc's fragile
      // handling of if-constexpr inside extended class lambdas.
      const bool pos_cached_rt = has_cacheable_positions_v<Coordinates>;

      // Two complete functors, selected by a HOST-level if constexpr. nvcc's extended-lambda
      // transformation miscompiles `if constexpr` INSIDE the lambda body here (wrong or
      // unregistered kernel stubs: cudaErrorInvalidResourceHandle, or a silently zero RHS), while
      // a lambda DEFINED inside an if-constexpr block is fine (the position fill above). So the
      // branch lives out here and each lambda body is branch-free.
      //
      // No explicit tile: Kokkos' own heuristic is used. An explicit tile matched to
      // DIFFRG_LAUNCH_BOUNDS was measured and is INERT (numtracer/gpubench/FINDINGS.md) -- the
      // 1.3-1.5x it appeared to give was GPU clock ramp, not tiling. Not worth the coupling: Kokkos
      // hard-aborts when the tile product exceeds LaunchBounds, so an explicit tile turns a future
      // launch-bounds change into a runtime abort.
      if constexpr (has_cacheable_positions_v<Coordinates>) {
        auto functor = KOKKOS_LAMBDA(const device::array<size_t, 1 + dim> &idx)
        {
          // make subview
          auto subview = device::apply([&](const auto &...i) { return Kokkos::subview(cache, i...); }, idx);

          // get the (precomputed) position for the current index
          device::array<pos_ctype, cdim> pos;
          for (size_t d = 0; d < cdim; ++d)
            pos[d] = static_cast<pos_ctype>(pos_view(idx[0] * cdim + d));

          device::array<ctype, dim> x;
          ctype weight = 1;
          for (int i = 0; i < dim; ++i) {
            x[i] = Kokkos::fma(scale[i], n[i][idx[1 + i]], start[i]);
            weight *= w[i][idx[1 + i]] * scale[i];
          }

          // Apply x/pos and the trailing arguments as two NESTED packs. Do not be tempted to
          // tuple_cat them into one tuple and apply that once: tuple_cat builds a by-value object,
          // and m_args holds every interpolator by value (device::make_tuple decays), so the
          // concatenated tuple is a full per-thread copy of all of them in local memory. That cost
          // 3576 B of stack frame on QCD_Nf2 (13 interpolators x 272 B) -- 89% of that binary's
          // 4000 B frame -- and 2.3x of runtime, measured. Applying an *lvalue* tuple binds
          // references instead, and KERNEL::kernel takes all of them by const&.
          // See docs/NUMTRACER_PER_THREAD_FRAME.md in the numtracer repo.
          device::apply(
              [&](const auto &...xargs) {
                device::apply([&](const auto &...iargs) { subview() = weight * KERNEL::kernel(xargs..., iargs...); },
                              m_args);
              },
              device::tuple_cat(x, pos));
        };
        Kokkos::parallel_for(make_kokkos_nd_range_divisible<1 + dim, ExecutionSpace>(space, {0}, extents),
                             KokkosNDLambdaWrapper<1 + dim, decltype(functor)>(functor));
      } else {
        auto functor = KOKKOS_LAMBDA(const device::array<size_t, 1 + dim> &idx)
        {
          // make subview
          auto subview = device::apply([&](const auto &...i) { return Kokkos::subview(cache, i...); }, idx);

          // get the position for the current index
          const auto idx_v = coordinates.from_linear_index(idx[0]);
          const auto pos = coordinates.forward(idx_v);

          device::array<ctype, dim> x;
          ctype weight = 1;
          for (int i = 0; i < dim; ++i) {
            x[i] = Kokkos::fma(scale[i], n[i][idx[1 + i]], start[i]);
            weight *= w[i][idx[1 + i]] * scale[i];
          }

          // Apply x/pos and the trailing arguments as two NESTED packs. Do not be tempted to
          // tuple_cat them into one tuple and apply that once: tuple_cat builds a by-value object,
          // and m_args holds every interpolator by value (device::make_tuple decays), so the
          // concatenated tuple is a full per-thread copy of all of them in local memory. That cost
          // 3576 B of stack frame on QCD_Nf2 (13 interpolators x 272 B) -- 89% of that binary's
          // 4000 B frame -- and 2.3x of runtime, measured. Applying an *lvalue* tuple binds
          // references instead, and KERNEL::kernel takes all of them by const&.
          // See docs/NUMTRACER_PER_THREAD_FRAME.md in the numtracer repo.
          device::apply(
              [&](const auto &...xargs) {
                device::apply([&](const auto &...iargs) { subview() = weight * KERNEL::kernel(xargs..., iargs...); },
                              m_args);
              },
              device::tuple_cat(x, pos));
        };
        Kokkos::parallel_for(make_kokkos_nd_range_divisible<1 + dim, ExecutionSpace>(space, {0}, extents),
                             KokkosNDLambdaWrapper<1 + dim, decltype(functor)>(functor));
      }

      using TeamType = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
      // reduction with vector lanes for warp-level parallelism
      constexpr int vector_width = 32;
      Kokkos::parallel_for(
          Kokkos::TeamPolicy(space, integral_view.size(), Kokkos::AUTO, vector_width),
          KOKKOS_CLASS_LAMBDA(const TeamType &team) {
            // get the current (continuous) index
            const uint k = team.league_rank();

            if (k >= integral_view.size()) return;

            // no-ops to capture
            (void)cache;
            (void)grid_size;

            // Flatten grid_size into total element count for thread+vector splitting
            size_t total_elements = 1;
            for (int d = 0; d < dim; ++d)
              total_elements *= grid_size[d];

            // Pre-compute stride array for index decomposition (avoids division/modulo in inner loop)
            device::array<size_t, dim> strides;
            strides[dim - 1] = 1;
            for (int d = dim - 2; d >= 0; --d)
              strides[d] = strides[d + 1] * grid_size[d + 1];

            NT res{};
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, (total_elements + vector_width - 1) / vector_width),
                [&](const size_t outer, NT &team_update) {
                  NT vec_sum{};
                  Kokkos::parallel_reduce(
                      Kokkos::ThreadVectorRange(team, vector_width),
                      [&](const size_t inner, NT &vec_update) {
                        const size_t flat = outer * vector_width + inner;
                        if (flat < total_elements) {
                          // Convert flat index back to multi-dimensional using pre-computed strides
                          device::array<size_t, dim> ridx;
                          size_t remainder = flat;
                          for (int d = 0; d < dim; ++d) {
                            ridx[d] = remainder / strides[d];
                            remainder -= ridx[d] * strides[d];
                          }
                          device::apply([&](const auto &...iargs) { vec_update += cache(k, iargs...); }, ridx);
                        }
                      },
                      vec_sum);
                  team_update += vec_sum;
                },
                res);

            // add the constant value (skip coordinate computation if kernel has no constant)
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              device::array<pos_ctype, cdim> pos;
              if (pos_cached_rt) {
                for (size_t d = 0; d < cdim; ++d)
                  pos[d] = static_cast<pos_ctype>(pos_view(size_t(k) * cdim + d));
              } else {
                pos = coordinates.forward(coordinates.from_linear_index(k));
              }
              // Nested packs, not tuple_cat -- same reason as the phase-1 functor above. This
              // kernel is only 0.3-4% of GPU time (docs/NUMTRACER_GPU_INVESTIGATION.md) but it
              // carried the whole per-thread copy too: 3544 B of stack frame on a kernel that does
              // almost no arithmetic.
              integral_view(k) =
                  res + device::apply(
                            [&](const auto &...pargs) {
                              return device::apply(
                                  [&](const auto &...iargs) { return KERNEL::constant(pargs..., iargs...); }, m_args);
                            },
                            pos);
            });
          });
    }

    /**
     * @brief Evaluate the integral at dest.size() points in a single launch: dest[i] is the integral with every
     * per-point argument taken at index i.
     *
     * Each argument is either one value shared by all points, or one value per point (a PointSpan, a
     * std::vector, or a PointArg holding either), so any argument of the kernel may vary between points. A
     * per-point argument must hold exactly dest.size() values. The nodes are summed on the fly, without an
     * intermediate (points x nodes) buffer.
     *
     * Rank-local: unlike map(), this is never split across MPI ranks and never touches MapScheduler,
     * so every rank may call it with its own number of points, also between collective operations.
     *
     * A single-precision integrator receives its arguments converted to single precision.
     */
    template <typename OT, typename... A>
      requires(
          internal::is_map_result<OT, NT> &&
          internal::accepts_args<NT, KERNEL, ctype, dim, internal::point_arg_value_t<A>...>)
    void map_points(const PointSpan<OT> dest, const A &...args)
    {
      run_map_points(dest, PointArg<internal::point_arg_value_t<A>>(args)...);
    }

    void set_map_points_policy(const MapPointsPolicy policy) { m_map_points_policy = policy; }

    /// Points evaluated per external grid point. Half of the scheduler's cost score.
    size_t quadrature_volume() const
    {
      size_t volume = 1;
      for (int i = 0; i < dim; ++i)
        volume *= grid_size[i];
      return volume;
    }

    template <typename OT, typename Coordinates, typename... Args>
      requires internal::is_map_result<OT, NT>
    auto map(OT *dest, const Coordinates &coordinates, const Args &...args)
    {
      internal::scheduled_map<ExecutionSpace>(integrator_id(), quadrature_volume(), dest, coordinates,
                                              [&](auto *d, const auto &c) { this->map_dist(d, c, args...); });
      return space;
    }

    /// map() without the MapScheduler: computes the whole of `coordinates` on this rank.
    template <typename OT, typename Coordinates, typename... Args>
    auto map_dist(OT *dest, const Coordinates &coordinates, const Args &...args)
    {
      internal::staged_map(space, m_staging.template get<OT>(), dest, coordinates.size(),
                           [this, coordinates, args...](const auto &view) {
                             this->map(this->space, view, coordinates, args...);
                           });
      return space;
    }

  private:
    template <typename OT, typename... T> void run_map_points(const PointSpan<OT> dest, const PointArg<T> &...args)
    {
      const size_t n = dest.size();
      if (n == 0) return;
      constexpr bool handles = internal::takes_handles<NT, KERNEL, ctype, dim, T...>;
      internal::check_kernel_precision<handles, ctype, T...>();
      const auto device_args = m_map_points.template stage<ctype, handles>(space, n, args...);
      m_map_points.run(space, dest, [&](const auto &result) { launch_map_points(result, n, device_args); });
    }

  public:
    /// The kernel launch of map_points(). Public only because nvcc rejects extended device lambdas
    /// inside non-public member functions.
    template <typename ResultView, typename DeviceArgs>
    void launch_map_points(const ResultView &result, const size_t n, const DeviceArgs &device_args)
    {
      using OT = typename ResultView::value_type;

      const auto &nd = nodes;
      const auto &w = weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;
      const auto gs = grid_size;
      const auto args = device_args;

      size_t total = 1;
      for (int d = 0; d < dim; ++d)
        total *= grid_size[d];

      const MapPointsPolicy policy = internal::resolve_policy(m_map_points_policy, n, total);

      if (policy == MapPointsPolicy::thread_per_point) {
        // Row-major node order and the constant added first, as in the serial TBBReduction path.
        Kokkos::parallel_for(
            "QuadratureIntegrator_map_points", Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
            KOKKOS_LAMBDA(const size_t i) {
              NT sum = device::apply([&](const auto &...a) { return NT(KERNEL::constant(a(i)...)); }, args);
              NT integral{};
              device::array<size_t, dim> idx{};
              for (size_t flat = 0; flat < total; ++flat) {
                device::array<ctype, dim> x;
                ctype weight = 1;
                for (int d = 0; d < dim; ++d) {
                  x[d] = Kokkos::fma(scale[d], nd[d][idx[d]], start[d]);
                  weight *= w[d][idx[d]] * scale[d];
                }
                device::apply(
                    [&](const auto &...xs) {
                      device::apply([&](const auto &...a) { integral += weight * KERNEL::kernel(xs..., a(i)...); },
                                    args);
                    },
                    x);
                for (int d = dim - 1; d >= 0; --d) {
                  if (++idx[d] < gs[d]) break;
                  idx[d] = 0;
                }
              }
              sum += integral;
              result(i) = static_cast<OT>(sum);
            });
      } else {
        using TeamType = typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;
        constexpr int vector_width = 32;
        const size_t n_outer = (total + vector_width - 1) / vector_width;

        device::array<size_t, dim> strides;
        strides[dim - 1] = 1;
        for (int d = dim - 2; d >= 0; --d)
          strides[d] = strides[d + 1] * grid_size[d + 1];

        Kokkos::parallel_for(
            "QuadratureIntegrator_map_points_team",
            Kokkos::TeamPolicy<ExecutionSpace>(space, n, Kokkos::AUTO, vector_width),
            KOKKOS_LAMBDA(const TeamType &team) {
              const size_t i = team.league_rank();
              NT integral{};
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadRange(team, n_outer),
                  [&](const size_t outer, NT &team_update) {
                    NT vec_sum{};
                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team, vector_width),
                        [&](const size_t inner, NT &vec_update) {
                          const size_t flat = outer * vector_width + inner;
                          if (flat >= total) return;
                          device::array<ctype, dim> x;
                          ctype weight = 1;
                          size_t remainder = flat;
                          for (int d = 0; d < dim; ++d) {
                            const size_t id = remainder / strides[d];
                            remainder -= id * strides[d];
                            x[d] = Kokkos::fma(scale[d], nd[d][id], start[d]);
                            weight *= w[d][id] * scale[d];
                          }
                          device::apply(
                              [&](const auto &...xs) {
                                device::apply(
                                    [&](const auto &...a) { vec_update += weight * KERNEL::kernel(xs..., a(i)...); },
                                    args);
                              },
                              x);
                        },
                        vec_sum);
                    team_update += vec_sum;
                  },
                  integral);
              Kokkos::single(Kokkos::PerTeam(team), [&]() {
                NT sum = device::apply([&](const auto &...a) { return NT(KERNEL::constant(a(i)...)); }, args);
                sum += integral;
                result(i) = static_cast<OT>(sum);
              });
            });
      }
    }

  protected:
    device::array<size_t, dim> grid_size;

    ExecutionSpace space;
    QuadratureProvider &quadrature_provider;
    device::array<device::array<ctype, dim>, 2> grid_extents;
    device::array<ctype, dim> grid_start;
    device::array<ctype, dim> grid_scale;

    device::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, dim> nodes;
    device::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, dim> weights;

    // Persistent view caches to avoid per-call GPU memory allocation
    mutable KokkosNDView<1 + dim, NT, ExecutionSpace> m_cache;
    mutable device::array<size_t, 1 + dim> m_cache_extents{};
    // Cached external positions for map(): one forward() per grid point instead of per thread.
    // Keyed on the coordinates' to_string() identity; flat layout [grid_point * cdim + d].
    mutable Kokkos::View<ctype *, typename ExecutionSpace::memory_space> m_positions;
    mutable std::string m_positions_key;
    mutable internal::MapStagingSet<NT, ExecutionSpace> m_staging;
    // map_points() buffers, separate from map()'s, whose staging may still be pending in a DeferredMaps scope.
    internal::MapPointsBuffers<NT, ExecutionSpace> m_map_points;
    MapPointsPolicy m_map_points_policy = MapPointsPolicy::automatic;
    mutable Kokkos::View<NT, typename ExecutionSpace::memory_space> m_result_view;
    mutable typename Kokkos::View<NT, typename ExecutionSpace::memory_space>::host_mirror_type m_result_host;
    mutable bool m_result_views_initialized = false;
  };

  template <int dim, typename NT, typename KERNEL>
  class QuadratureIntegrator<dim, NT, KERNEL, TBB_exec> : public QuadratureIntegrator<dim, NT, KERNEL, KokkosHost_exec>
  {
    using Base = QuadratureIntegrator<dim, NT, KERNEL, KokkosHost_exec>;

  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or
     * possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = TBB_exec;

    QuadratureIntegrator(QuadratureProvider &quadrature_provider, const std::array<size_t, dim> _grid_size,
                         std::array<ctype, dim> grid_min, std::array<ctype, dim> grid_max,
                         const std::array<QuadratureType, dim> quadrature_type)
        : Base(quadrature_provider, _grid_size, grid_min, grid_max, quadrature_type)
    {
    }

    template <typename... T>
      requires internal::accepts_args<NT, KERNEL, ctype, dim, T...>
    void get(NT &dest, const T &...t) const
    {
      // A single-precision integrator evaluates its kernel in single precision, and a kernel that
      // takes handles gets those.
      constexpr bool handles = internal::takes_handles<NT, KERNEL, ctype, dim, T...>;
      internal::check_kernel_precision<handles, ctype, T...>();
      if constexpr (!(std::is_same_v<T, internal::launch_arg_t<handles, T, ctype>> && ...))
        get(dest, internal::to_launch_arg<handles, ctype>(t)...);
      else {
        const auto args = device::tie(t...);

        const auto &n = nodes;
        const auto &w = weights;
        const auto &start = grid_start;
        const auto &scale = grid_scale;

        auto functor = [&](const device::array<size_t, dim> &idx) {
          device::array<ctype, dim> x;
          ctype weight = 1;
          for (size_t i = 0; i < dim; ++i) {
            x[i] = Kokkos::fma(scale[i], n[i][idx[i]], start[i]);
            weight *= w[i][idx[i]] * scale[i];
          }
          return device::apply([&](const auto &...iargs) { return weight * KERNEL::kernel(iargs...); },
                               device::tuple_cat(x, args));
        };

        dest = KERNEL::constant(t...) + TBBReduction<dim, NT, decltype(functor)>(grid_size, functor);
      }
    }

    template <typename OT, typename... T>
      requires(internal::is_widened_result<OT, NT> &&
               internal::accepts_args<NT, KERNEL, ctype, dim, T...>)
    void get(OT &dest, const T &...t) const
    {
      NT result;
      get(result, t...);
      dest = OT(result);
    }

    /// See QuadratureIntegrator::map_points. One get() per point inside a flat parallel loop, so every
    /// result is bitwise identical to the corresponding get().
    template <typename OT, typename... A>
      requires(
          internal::is_map_result<OT, NT> &&
          internal::accepts_args<NT, KERNEL, ctype, dim, internal::point_arg_value_t<A>...>)
    void map_points(const PointSpan<OT> dest, const A &...args) const
    {
      internal::map_points_by_get(*this, dest, PointArg<internal::point_arg_value_t<A>>(args)...);
    }

    template <typename OT, typename Coordinates, typename... Args>
      requires internal::is_map_result<OT, NT>
    void map(execution_space &, OT *dest, const Coordinates &coordinates, const Args &...args)
    {
      // Handles once here rather than per point in get().
      constexpr bool handles = internal::map_takes_handles<NT, KERNEL, ctype, dim, Coordinates, Args...>;
      internal::check_kernel_precision<handles, ctype, Args...>();
      const auto m_args = [&] {
        if constexpr (handles)
          return device::make_tuple(DiFfRG::to_kernel_handle<ctype>(args)...);
        else
          return device::tie(args...);
      }();

      tbb::parallel_for(tbb::blocked_range<uint>(0, coordinates.size()), [&](const tbb::blocked_range<uint> &r) {
        for (uint idx = r.begin(); idx != r.end(); ++idx) {
          const auto dis_idx = coordinates.from_linear_index(idx);
          const auto pos = coordinates.forward(dis_idx);
          // nested packs rather than tuple_cat, which would copy every argument per point
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
          this->integrator_id(), Base::quadrature_volume(), dest, coordinates, [&](auto *d, const auto &c) {
            // tbb::parallel_for writes straight into `dest`, so there is nothing to stage.
            internal::run_or_queue_host([this, d, c, args...]() {
              auto sp = execution_space();
              this->map(sp, d, c, args...);
            });
            if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
          });
      return execution_space();
    }

  protected:
    using Base::grid_extents;
    using Base::grid_scale;
    using Base::grid_size;
    using Base::grid_start;
    using Base::quadrature_provider;

    using Base::nodes;
    using Base::weights;
  };
} // namespace DiFfRG
