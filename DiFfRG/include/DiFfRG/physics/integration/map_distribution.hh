#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/multi_value.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/map_completion.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>

// std
#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

namespace DiFfRG
{
  namespace internal
  {
    /// Grow-only result buffers of map(): device scratch, plus page-locked staging so the copy back
    /// is genuinely asynchronous (see MapCompletion).
    template <typename T, typename ExecutionSpace> struct MapStaging {
      Kokkos::View<T *, ExecutionSpace> device;
      size_t device_size = 0;
      Kokkos::View<T *, PinnedHost_memory> pinned;
      size_t pinned_size = 0;

      auto device_view(const ExecutionSpace &space, const size_t n)
      {
        if (device_size < n) {
          device = Kokkos::View<T *, ExecutionSpace>(Kokkos::view_alloc(space, "MapIntegrators_device_view"), n);
          device_size = n;
        }
        return Kokkos::View<T *, ExecutionSpace>(device, Kokkos::make_pair(size_t(0), n));
      }

      auto pinned_view(const size_t n)
      {
        if (pinned_size < n) {
          pinned = Kokkos::View<T *, PinnedHost_memory>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing, "MapIntegrators_pinned_view"), n);
          pinned_size = n;
        }
        return Kokkos::View<T *, PinnedHost_memory>(pinned, Kokkos::make_pair(size_t(0), n));
      }
    };

    /// map() result buffers for value type NT and, when that is single precision, for its double
    /// counterpart. The kernel writes each result straight into the destination-typed buffer, so a
    /// float integrator mapping into double converts on the device.
    template <typename NT, typename ExecutionSpace> struct MapStagingSet {
      MapStaging<NT, ExecutionSpace> native;
      MapStaging<get_type::double_precision<NT>, ExecutionSpace> widened;

      template <typename OT> auto &get()
      {
        if constexpr (std::is_same_v<OT, NT>)
          return native;
        else
          return widened;
      }
    };

    /**
     * @brief Run a synchronous host map now, or queue it for flush time if a deferral scope is open.
     *
     * A host kernel blocks the host thread, and with it the launch of any device map issued after
     * it. Queuing makes the call order inside a DeferredMaps scope irrelevant: flush() runs the queue
     * before it fences, i.e. while the device work already launched is still running. See
     * MapCompletion::record_work. Compiled out in a CUDA-less build, where there is no device to
     * overlap with.
     *
     * `job` must own copies of everything except the integrator and the destination, which the
     * DeferredMaps contract already requires to outlive the scope.
     */
    template <typename Job> void run_or_queue_host(Job &&job)
    {
      if constexpr (has_device_backend) {
        if (MapCompletion::deferral_enabled()) {
          MapCompletion::record_work(std::forward<Job>(job));
          return;
        }
      }
      job();
    }

    /// The destination arrays of a map(): one pointer, or one per component of a MultiValue result.
    template <typename OT> std::array<void *, 1> dest_pointers(OT *dest) { return {dest}; }
    template <typename OT, size_t N> std::array<void *, N> dest_pointers(const std::array<OT *, N> &dest)
    {
      std::array<void *, N> p;
      for (size_t j = 0; j < N; ++j)
        p[j] = dest[j];
      return p;
    }

    /// The destination of a map() advanced by @p offset elements.
    template <typename OT> OT *offset_dest(OT *dest, const size_t offset) { return dest + offset; }
    template <typename OT, size_t N> std::array<OT *, N> offset_dest(std::array<OT *, N> dest, const size_t offset)
    {
      for (auto &d : dest)
        d += offset;
      return dest;
    }

    /// What one point of a map() destination holds: `OT` for `OT *`, `MultiValue<OT, N>` for N pointers.
    template <typename Dest> struct map_dest;
    template <typename OT> struct map_dest<OT *> {
      using result = OT;
      static constexpr size_t elem_size = sizeof(OT);
    };
    template <typename OT, size_t N> struct map_dest<std::array<OT *, N>> {
      using result = MultiValue<OT, N>;
      static constexpr size_t elem_size = sizeof(OT);
    };
    template <typename Dest> using map_dest_result_t = typename map_dest<Dest>::result;

    /// Write the result @p v of point @p i to its destination.
    template <typename OT> void store_result(OT *dest, const size_t i, const OT &v) { dest[i] = v; }
    template <typename OT, size_t N>
    void store_result(const std::array<OT *, N> &dest, const size_t i, const MultiValue<OT, N> &v)
    {
      for (size_t j = 0; j < N; ++j)
        dest[j][i] = v[j];
    }

    /**
     * @brief Cost of a map() with @p n outputs relative to one with a single output, for MapScheduler.
     *
     * Each extra projection of the same loop measured +25--30 % on an RTX 4070 (lambda4L groups of 1--3), not
     * +100 %: the shared part of the integrand dominates.
     */
    inline double multi_output_cost(const size_t n) { return 1. + 0.27 * double(n - 1); }

    /**
     * @brief The MapScheduler half of every integrator's map().
     *
     * Registers the call with the scheduler and hands this rank's slice of the external grid to
     * `run(dest, coordinates)` -- either the whole grid or a SubCoordinates window of it, with `dest`
     * offset to match. `run` is responsible for landing (or deferring) its own result.
     *
     * `dest` is a pointer, or a std::array of pointers for a MultiValue result; every array receives
     * the same slice.
     *
     * Every rank calls this for every map(), owner or not; see MapScheduler::schedule for why that
     * is what keeps the collective matched.
     */
    template <typename ExecutionSpace, typename Dest, typename Coordinates, typename Run>
    void scheduled_map(const size_t integrator_id, const size_t quadrature_volume, const Dest &dest,
                       const Coordinates &coordinates, Run &&run)
    {
      auto &scheduler = MapScheduler::instance();

      // One staging buffer per integrator, so a second map() from this integrator before a flush
      // would clobber the first result. The *plan* is what decides, not the local pending list: a
      // rank that owned no slice of the earlier map has nothing pending and would skip a flush the
      // other ranks perform, leaving the collectives mismatched.
      if (scheduler.active() && scheduler.plan_contains(integrator_id)) MapCompletion::flush();

      const auto dests = dest_pointers(dest);
      const auto weighted_volume =
          static_cast<size_t>(std::llround(double(quadrature_volume) * multi_output_cost(dests.size())));
      const MapSlice slice =
          scheduler.schedule(integrator_id, std::span<void *const>(dests), map_dest<Dest>::elem_size, coordinates.size(),
                             weighted_volume, /* splittable */ true, map_target<ExecutionSpace>());

      if (slice.count == 0) {
        // Not an owner: no kernels, but the plan entry is registered, so this rank still has to
        // take part in the exchange.
        if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
        return;
      }
      if (slice.owns_all(coordinates.size()))
        run(dest, coordinates);
      else
        run(offset_dest(dest, slice.offset), SubCoordinates(coordinates, slice.offset, slice.count));
    }

    /**
     * @brief The unscheduled body of map(): `launch(view)` fills a device view of n results, which
     * then has to reach `dest`.
     *
     * On a device the copy back goes through page-locked staging, so it is genuinely asynchronous,
     * and inside a DeferredMaps scope the result stays there until the scope's flush lands it. On a
     * host backend there is nothing to stage, but the work is synchronous, so inside a deferral scope
     * it is queued instead (run_or_queue_host).
     *
     * Outside a deferral scope `dest` is valid on return: flush() fences, lands the
     * staged copy and -- under MPI -- exchanges this batch's slices.
     *
     * `launch` is kept by value in a queued host job, so it must capture its arguments by value.
     */
    template <typename ExecutionSpace, typename OT, typename Launch>
    void staged_map(ExecutionSpace &space, MapStaging<OT, ExecutionSpace> &stage, OT *dest, const size_t n,
                    Launch launch)
    {
      if constexpr (std::is_same_v<typename ExecutionSpace::memory_space, CPU_memory>) {
        // Queued jobs run one after another, so the shared device scratch is written and drained
        // before the next job touches it.
        run_or_queue_host([&space, &stage, dest, n, launch]() {
          auto device_view = stage.device_view(space, n);
          launch(device_view);
          Kokkos::deep_copy(space, Kokkos::View<OT *, CPU_memory, Kokkos::MemoryUnmanaged>(dest, n), device_view);
        });
      } else {
        // Land an outstanding result first; in the normal call pattern (each flow mapped once per
        // flush interval) this never triggers.
        if (stage.pinned_size > 0 && MapCompletion::has_pending(stage.pinned.data())) MapCompletion::flush();

        auto device_view = stage.device_view(space, n);
        auto pinned_view = stage.pinned_view(n);
        launch(device_view);

        // Copying straight into `dest` -- ordinary pageable caller memory, e.g. a dealii::Vector
        // element range -- would block until the kernels feeding it have finished, leaving the host
        // no run-ahead at all. See MapCompletion for the measurement.
        Kokkos::deep_copy(space, pinned_view, device_view);
        MapCompletion::record(dest, stage.pinned.data(), n * sizeof(OT));
      }
      if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
    }

    /**
     * @brief staged_map() for a MultiValue result: component j of the n results lands in dest[j].
     *
     * The device writes the results interleaved; the copy into the destinations is a strided
     * pending copy per component, landed at flush time like a plain map() result.
     */
    template <typename ExecutionSpace, typename OT, size_t N, typename Launch>
    void staged_map(ExecutionSpace &space, MapStaging<MultiValue<OT, N>, ExecutionSpace> &stage,
                    const std::array<OT *, N> &dest, const size_t n, Launch launch)
    {
      using MV = MultiValue<OT, N>;
      static_assert(sizeof(MV) == N * sizeof(OT), "MultiValue must be laid out as a plain array");
      if constexpr (std::is_same_v<typename ExecutionSpace::memory_space, CPU_memory>) {
        run_or_queue_host([&space, &stage, dest, n, launch]() {
          auto device_view = stage.device_view(space, n);
          launch(device_view);
          space.fence();
          for (size_t i = 0; i < n; ++i)
            store_result(dest, i, device_view(i));
        });
      } else {
        if (stage.pinned_size > 0 && MapCompletion::has_pending(stage.pinned.data())) MapCompletion::flush();

        auto device_view = stage.device_view(space, n);
        auto pinned_view = stage.pinned_view(n);
        launch(device_view);
        Kokkos::deep_copy(space, pinned_view, device_view);
        const char *pinned = reinterpret_cast<const char *>(stage.pinned.data());
        for (size_t j = 0; j < N; ++j)
          MapCompletion::record_strided(dest[j], pinned + j * sizeof(OT), n, sizeof(OT), sizeof(MV));
      }
      if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
    }
  } // namespace internal
} // namespace DiFfRG
