#pragma once

#include <DiFfRG/common/tuples.hh>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <type_traits>
#include <vector>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#else
#include <array>
#include <tuple>
#include <utility>
#endif

namespace DiFfRG
{
  /**
   * @brief The CPU execution space: TBB, the one host thread pool DiFfRG runs on.
   *
   * Not a Kokkos backend but a tag that routes an integrator through tbb::parallel_for instead, so
   * that its work is stolen by the same arena the assemblers' parallel loops use. This is what every
   * production flow's CPU integrators are instantiated with, and it is optimal in conjunction with
   * the discretizations, whose cell loops it nests inside.
   *
   * Its sibling KokkosHost_exec is the *Kokkos* host backend, which is Kokkos::Serial unless the
   * build sets KOKKOS_THREADS=ON. Keeping it serial is deliberate: a second pool of spinning
   * workers on the same cores only contends with this one.
   */
  struct TBB_ExecutionSpace {
    using memory_space = Kokkos::DefaultHostExecutionSpace::memory_space;
    void fence() const {}
  };

  class ExecutionSpaces
  {
  public:
    using GPU_exec_space = Kokkos::DefaultExecutionSpace;
    using GPU_memory_space = GPU_exec_space::memory_space;

    using KokkosHost_exec_space = Kokkos::DefaultHostExecutionSpace;
    using KokkosHost_memory_space = KokkosHost_exec_space::memory_space;

    using TBB_exec_space = TBB_ExecutionSpace;
    using TBB_memory_space = TBB_exec_space::memory_space;
  };

  using GPU_memory = ExecutionSpaces::GPU_memory_space;
  using KokkosHost_memory = ExecutionSpaces::KokkosHost_memory_space;
  using TBB_memory = ExecutionSpaces::TBB_memory_space;

  using CPU_memory = Kokkos::DefaultHostExecutionSpace::memory_space;

  /**
   * @brief Host memory the device can DMA to/from without staging, i.e. page-locked.
   *
   * A device-to-*pageable*-host `cudaMemcpyAsync` is synchronous in practice, because the driver
   * must bounce it through its own pinned buffer; copying into this space instead is what makes an
   * asynchronous result copy actually asynchronous. Falls back to ordinary host memory when there
   * is no CUDA backend, where the distinction does not exist.
   */
#ifdef KOKKOS_ENABLE_CUDA
  using PinnedHost_memory = Kokkos::CudaHostPinnedSpace;
#else
  using PinnedHost_memory = CPU_memory;
#endif

  using GPU_exec = ExecutionSpaces::GPU_exec_space;
  using KokkosHost_exec = ExecutionSpaces::KokkosHost_exec_space;
  using TBB_exec = ExecutionSpaces::TBB_exec_space;

  // Ensure that CPU memory space is the same as the Kokkos host and TBB memory spaces.
  // We assume that this is true, and when switching to a different memory space, it is always unique.
  static_assert(std::is_same_v<CPU_memory, KokkosHost_memory>,
                "CPU memory space must be the same as the Kokkos host memory space");
  static_assert(std::is_same_v<CPU_memory, TBB_memory>, "CPU memory space must be the same as TBB memory space");

  /**
   * @brief An extension of the Kokkos::Sum reducer that adds a constant value to the result.
   *
   * @tparam Scalar the type of the scalar value to be summed.
   * @tparam Space execution space for the Kokkos::Sum reducer.
   */
  template <class Scalar, class SavedScalar, class Space> struct SumPlus {
  public:
    // Required
    using reducer = SumPlus<Scalar, SavedScalar, Space>;
    using value_type = std::remove_cv_t<Scalar>;
    using saved_type = std::remove_cv_t<SavedScalar>;
    static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

    using result_view_type = Kokkos::View<value_type, Space>;

  private:
    result_view_type value;
    bool references_scalar_v;

    const saved_type plus_value;

  public:
    KOKKOS_INLINE_FUNCTION
    SumPlus(value_type &value_, const saved_type &plus_value_)
        : value(&value_), references_scalar_v(true), plus_value(plus_value_)
    {
    }

    KOKKOS_INLINE_FUNCTION
    SumPlus(const result_view_type &value_, const saved_type &plus_value_)
        : value(value_), references_scalar_v(false), plus_value(plus_value_)
    {
    }

    // Required
    KOKKOS_INLINE_FUNCTION
    void join(value_type &dest, const value_type &src) const { dest += src; }

    KOKKOS_INLINE_FUNCTION
    void init(value_type &val) const { val = Kokkos::reduction_identity<value_type>::sum(); }

    KOKKOS_INLINE_FUNCTION
    value_type &reference() const { return *value.data(); }

    KOKKOS_INLINE_FUNCTION
    result_view_type view() const { return value; }

    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const { return references_scalar_v; }

    KOKKOS_INLINE_FUNCTION
    void final(value_type &update) const { update += plus_value; }
  };

  namespace device
  {
#ifdef KOKKOS_ENABLE_CUDA
    template <typename... T> using tuple = cuda::std::tuple<T...>;
    template <typename T, std::size_t N> using array = cuda::std::array<T, N>;
    using cuda::std::apply;
    using cuda::std::forward;
    using cuda::std::forward_as_tuple;
    using cuda::std::get;
    using cuda::std::index_sequence;
    using cuda::std::integer_sequence;
    using cuda::std::make_integer_sequence;
    using cuda::std::make_tuple;
    using cuda::std::tie;
    using cuda::std::tuple_cat;
    using cuda::std::tuple_element;
#else
    template <typename... T> using tuple = std::tuple<T...>;
    template <typename T, std::size_t N> using array = std::array<T, N>;
    using std::apply;
    using std::forward;
    using std::forward_as_tuple;
    using std::get;
    using std::index_sequence;
    using std::integer_sequence;
    using std::make_integer_sequence;
    using std::make_tuple;
    using std::tie;
    using std::tuple_cat;
    using std::tuple_element;
#endif
  } // namespace device

  template <int dim, typename T> struct GetKokkosNDStarType {
    using type = typename GetKokkosNDStarType<dim - 1, T>::type *;
  };
  template <typename T> struct GetKokkosNDStarType<1, T> {
    using type = T *;
  };

  // ------------------------------------------------
  // Getting View types
  // ------------------------------------------------

  template <int dim, typename T, typename ExecutionSpace>
  using KokkosNDView = Kokkos::View<typename GetKokkosNDStarType<dim, T>::type, // Get the star syntax for
                                                                                // dimensionality recursively with
                                    ExecutionSpace                              // Choice between GPU and CPU
                                    >;
  template <int dim, typename T, typename ExecutionSpace>
  using KokkosNDViewRestrict =
      Kokkos::View<typename GetKokkosNDStarType<dim, T>::type, // Get the star syntax for dimensionality recursively
                                                               // with a helper
                   ExecutionSpace,                             // Choice between GPU and CPU
                   Kokkos::MemoryTraits<Kokkos::Restrict>      // No-alias hint for compiler optimization
                   >;

  template <int dim, typename T, typename ExecutionSpace>
  using KokkosNDViewUnmanaged =
      Kokkos::View<typename GetKokkosNDStarType<dim, T>::type, // Get the star syntax for dimensionality recursively
                                                               // with a helper
                   ExecutionSpace,                             // Choice between GPU and CPU
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>     // No allocation: Attach to existing memory
                   >;

  template <int dim, typename T, typename ExecutionSpace>
  auto make_kokkos_nd_view(const std::string &label, const device::array<size_t, dim> &extents)
  {
    return device::apply([&](const auto &...args) { return KokkosNDView<dim, T, ExecutionSpace>(label, args...); },
                         extents);
  }

  template <int dim, typename T, typename ExecutionSpace>
  auto make_kokkos_nd_view_restrict(const std::string &label, const device::array<size_t, dim> &extents)
  {
    return device::apply(
        [&](const auto &...args) { return KokkosNDViewRestrict<dim, T, ExecutionSpace>(label, args...); }, extents);
  }

  // ------------------------------------------------
  // Getting ranges to iterate over
  // ------------------------------------------------
  // Pin launch bounds. Without __launch_bounds__ ptxas cannot see the block size and pins registers
  // differently; measured ~3-14% slower across the YangMills flow set on sm_89, results identical
  // (numtracer/gpubench/FINDINGS.md). Override with -DDIFFRG_LAUNCH_BOUNDS=N.
#ifndef DIFFRG_LAUNCH_BOUNDS
#define DIFFRG_LAUNCH_BOUNDS 128
#endif
#if DIFFRG_LAUNCH_BOUNDS == 0 // A/B control: no launch bounds at all (the historical behaviour)
  template <int dim, typename ExecutionSpace> struct KokkosNDRangeHelper {
    using type = Kokkos::MDRangePolicy<Kokkos::Rank<dim>, ExecutionSpace>;
  };
  template <typename ExecutionSpace> struct KokkosNDRangeHelper<1, ExecutionSpace> {
    using type = Kokkos::RangePolicy<ExecutionSpace>;
  };
#else
  template <int dim, typename ExecutionSpace> struct KokkosNDRangeHelper {
    using type = Kokkos::MDRangePolicy<Kokkos::Rank<dim>, ExecutionSpace, Kokkos::LaunchBounds<DIFFRG_LAUNCH_BOUNDS>>;
  };
  template <typename ExecutionSpace> struct KokkosNDRangeHelper<1, ExecutionSpace> {
    using type = Kokkos::RangePolicy<ExecutionSpace, Kokkos::LaunchBounds<DIFFRG_LAUNCH_BOUNDS>>;
  };
#endif
  template <int dim, typename ExecutionSpace> using KokkosNDRange = KokkosNDRangeHelper<dim, ExecutionSpace>::type;

  // Use a tile whose every dimension DIVIDES its extent, instead of Kokkos' heuristic.
  //
  // Kokkos picks a tile by halving its recommended tile until the product fits under
  // min(512, LaunchBounds::maxTperB) -- so every tile dimension it can produce is the recommendation
  // divided by a power of two. When an extent is not a multiple of that, the boundary tile launches
  // masked-off lanes that occupy warp slots and do no work.
  //
  // That is not hypothetical here. Every QCD_Nf2 model uses angular order 6, and the shipped
  // LB=128 gives tile {16,2,4,1,1} on the rank-5 flows: extent 6 against tile 4 is ceil(6/4)=2 tiles
  // covering 8 slots, i.e. 4/3 of the threads launched for 1 unit of work. Since dim0 x dim1 = 32 is
  // exactly one warp, the masked lanes form whole dead warps. Measured masked fractions at LB=128
  // (numtracer/gpubench/tools/tilewaste.py): vacuum no_mesons 1.255x, with_mesons 1.306x,
  // with_mesons_3D 1.299x, finite_T no_mesons 1.314x -- and YangMills 1.000x, because its angular
  // orders are 8, which the power-of-two tile does divide.
  //
  // This is what the A100 LaunchBounds sweep actually measured. LB=96 and LB=64 both collapse the
  // tile to {16,2,2,1,1}, which divides 6 -- and LB=64 came out 8.0% faster than LB=128 at
  // *byte-identical* registers (255), spill (1864 B) and occupancy (12.5%), which no occupancy story
  // explains. Per-flow it splits perfectly: the three rank-5 flows (the only ones with masked lanes)
  // gained 21.7-24.2% against 25.0% predicted from the masked fraction alone, while every rank-4
  // flow -- with nothing to recover -- got 0-21.6% *slower* from the smaller block.
  //
  // Set -DDIFFRG_DIVISIBLE_TILE=0 to restore Kokkos' heuristic (the A/B control).
#ifndef DIFFRG_DIVISIBLE_TILE
#define DIFFRG_DIVISIBLE_TILE 1
#endif

  /**
   * @brief A tile whose every dimension divides its extent, so no lane is launched masked.
   *
   * @param extents      iteration-space extents
   * @param kokkos_tile  the tile Kokkos would have chosen; used as the seed and as the tie-breaker,
   *                     so this function inherits Kokkos' architecture-specific recommendation
   *                     rather than duplicating it (which would silently drift on a Kokkos bump)
   * @param budget       maximum tile product, i.e. the block size ceiling
   *
   * Among all tiles that divide their extents and fit the budget, take the largest product -- a
   * bigger block means fewer blocks and less of the register file lost to per-warp allocation
   * granularity -- breaking ties toward Kokkos' shape, which encodes the coalescing preference on
   * dim 0. Returns `kokkos_tile` unchanged when it already divides everything, so architectures and
   * models that were already clean (e.g. YangMills, whose orders are 8) are untouched.
   */
  template <int dim>
  device::array<size_t, dim> compute_divisible_tile(const device::array<size_t, dim> &extents,
                                                    const device::array<size_t, dim> &kokkos_tile, size_t budget)
  {
    bool already_divides = true;
    for (int i = 0; i < dim; ++i)
      if (kokkos_tile[i] == 0 || extents[i] % kokkos_tile[i] != 0) already_divides = false;
    if (already_divides) return kokkos_tile;

    // Candidate tile values per dimension: the divisors of the extent that fit the budget. Extents
    // here are quadrature orders and grid sizes (order 10-100), so these lists are short and the
    // search below visits a few thousand nodes at worst -- negligible against a millisecond kernel.
    std::array<std::vector<size_t>, dim> candidates;
    for (int i = 0; i < dim; ++i) {
      const size_t cap = std::min<size_t>(extents[i], budget);
      for (size_t d = 1; d <= cap; ++d)
        if (extents[i] % d == 0) candidates[i].push_back(d);
      if (candidates[i].empty()) candidates[i].push_back(1); // extent 0: degenerate, launch nothing
    }

    // Ranking, in order:
    //  (1) whole warps -- a block that is not a multiple of 32 pads its last warp with dead lanes,
    //      which is the very defect being fixed, only at warp instead of tile granularity;
    //  (2) largest block -- fewer blocks, and the register file is handed out per warp;
    //  (3) closest to Kokkos' shape, measured as sum |log2(tile/kokkos)| so the metric is
    //      scale-free: 16->64 costs 2 and 4->2 costs 1, which stops the search from collapsing the
    //      angular dimensions to 1 just to inflate dim 0;
    //  (4) tile weight as far forward as possible, since dim 0 is the contiguous one under
    //      LayoutLeft and a wider dim-0 tile coalesces the cache write better.
    constexpr size_t warp = 32;
    const auto log_distance = [&](const device::array<size_t, dim> &tile) {
      double d = 0.;
      for (int i = 0; i < dim; ++i)
        d += std::abs(std::log2(static_cast<double>(tile[i]) / static_cast<double>(kokkos_tile[i])));
      return d;
    };
    const auto lexicographically_greater = [](const device::array<size_t, dim> &a,
                                              const device::array<size_t, dim> &b) {
      for (int i = 0; i < dim; ++i)
        if (a[i] != b[i]) return a[i] > b[i];
      return false;
    };

    device::array<size_t, dim> best{}, current{};
    size_t best_product = 0;
    double best_distance = 0.;
    bool best_whole_warps = false, have_best = false;

    const auto visit = [&](auto &&self, int i, size_t product) -> void {
      if (i == dim) {
        const bool whole_warps = (product % warp == 0);
        const double distance = log_distance(current);
        bool better;
        if (!have_best)
          better = true;
        else if (whole_warps != best_whole_warps)
          better = whole_warps;
        else if (product != best_product)
          better = product > best_product;
        else if (std::abs(distance - best_distance) > 1e-9)
          better = distance < best_distance;
        else
          better = lexicographically_greater(current, best);
        if (better) {
          best = current;
          best_product = product;
          best_distance = distance;
          best_whole_warps = whole_warps;
          have_best = true;
        }
        return;
      }
      for (const size_t d : candidates[i]) {
        if (product * d > budget) break; // candidates are ascending, so nothing further fits either
        current[i] = d;
        self(self, i + 1, product * d);
      }
    };
    visit(visit, 0, 1);

    return have_best ? best : kokkos_tile;
  }

  /**
   * @brief Compute clamped tile sizes for MDRangePolicy so that the product of tile dimensions does not exceed
   * max_threads. Fills from the innermost (last) dimension outward.
   */
  template <int dim>
  device::array<size_t, dim> compute_tile_hints(const device::array<size_t, dim> &extents, size_t max_threads = 256)
  {
    device::array<size_t, dim> tile;
    for (int i = 0; i < dim; ++i)
      tile[i] = 1;

    size_t budget = max_threads;
    // Fill from innermost dimension outward
    for (int i = dim - 1; i >= 0; --i) {
      tile[i] = std::min(extents[i], budget);
      budget /= tile[i];
      if (budget == 0) break;
    }
    return tile;
  }

  template <int dim, typename ExecutionSpace>
  auto make_kokkos_nd_range(ExecutionSpace &space, const device::array<size_t, dim> start,
                            const device::array<size_t, dim> end)
  {
    if constexpr (dim == 1) {
      return KokkosNDRange<dim, ExecutionSpace>(space, start[0], end[0]);
    } else {
      Kokkos::Array<size_t, dim> start_view;
      Kokkos::Array<size_t, dim> end_view;
      for (size_t i = 0; i < dim; ++i) {
        start_view[i] = start[i];
        end_view[i] = end[i];
      }
      return KokkosNDRange<dim, ExecutionSpace>(space, start_view, end_view);
    }
  }

  template <int dim, typename ExecutionSpace>
  auto make_kokkos_nd_range(ExecutionSpace &space, const device::array<size_t, dim> start,
                            const device::array<size_t, dim> end, const device::array<size_t, dim> tile)
  {
    if constexpr (dim == 1) {
      return KokkosNDRange<dim, ExecutionSpace>(space, start[0], end[0]);
    } else {
      Kokkos::Array<size_t, dim> start_view;
      Kokkos::Array<size_t, dim> end_view;
      Kokkos::Array<size_t, dim> tile_view;
      for (size_t i = 0; i < dim; ++i) {
        start_view[i] = start[i];
        end_view[i] = end[i];
        tile_view[i] = tile[i];
      }
      return KokkosNDRange<dim, ExecutionSpace>(space, start_view, end_view, tile_view);
    }
  }

  /**
   * @brief Like `make_kokkos_nd_range`, but re-tiled so no lane is launched masked.
   *
   * Builds the policy once with Kokkos' own tiling to read back what it chose (`m_tile`), then
   * rebuilds with a divisibility-corrected tile of at most the same product. Because the product
   * never grows, this cannot trip Kokkos' "tile dimensions exceed LaunchBounds" abort
   * (KokkosExp_MDRangePolicy.hpp:449-462).
   *
   * Only for `parallel_for` over a write-per-thread range. Do NOT use it on the `parallel_reduce`
   * paths: there the tile shape sets the reduction tree, so re-tiling would not be bit-identical.
   */
  template <int dim, typename ExecutionSpace>
  auto make_kokkos_nd_range_divisible(ExecutionSpace &space, const device::array<size_t, dim> start,
                                      const device::array<size_t, dim> end)
  {
    auto policy = make_kokkos_nd_range<dim, ExecutionSpace>(space, start, end);
#if DIFFRG_DIVISIBLE_TILE
    if constexpr (dim > 1) {
      device::array<size_t, dim> extents, kokkos_tile;
      size_t budget = 1;
      bool changed = false;
      for (int i = 0; i < dim; ++i) {
        extents[i] = end[i] - start[i];
        kokkos_tile[i] = static_cast<size_t>(policy.m_tile[i]);
        budget *= kokkos_tile[i];
      }
      const auto tile = compute_divisible_tile<dim>(extents, kokkos_tile, budget);
      for (int i = 0; i < dim; ++i)
        changed |= (tile[i] != kokkos_tile[i]);
      if (changed) return make_kokkos_nd_range<dim, ExecutionSpace>(space, start, end, tile);
    }
#endif
    return policy;
  }

  template <int dim, typename TeamType>
  KOKKOS_FORCEINLINE_FUNCTION auto make_kokkos_nd_thread_range(const TeamType &team,
                                                               const device::array<size_t, dim> end)
  {
    if constexpr (dim == 1) {
      return Kokkos::TeamThreadRange(team, end[0]);
    } else {
      return device::apply([&](const auto &...args) { return Kokkos::TeamThreadMDRange(team, args...); }, end);
    }
  }

  // ------------------------------------------------
  // Wrap Kokkos lambdas
  // ------------------------------------------------

  /**
   * @brief This is a functor which wraps a lambda.
   * Basically, this is necessary when one wants to call a variadic lambda on an NVIDIA GPU.
   * CUDA seems to be unable to expand the variadic arguments - in contrast, a direct approach does indeed work for
   * openMP or serial compilation.
   * To get around this limitation, the KokkosNDLambdaWrapper packs the indices into an array.
   * If you wonder, whether there's a difference when using tie and tuples: https://godbolt.org/z/M3bG39rsM
   * No. Therefore, we spare the ourselves the hassle and simply use an array.
   *
   * @tparam dim Number of arguments taken
   * @tparam FUN The lambda to which we forward the indices
   */
  template <int dim, typename FUN> struct KokkosNDLambdaWrapper {
    KOKKOS_FUNCTION
    KokkosNDLambdaWrapper(const FUN &_fun) : fun(_fun) {};

    template <typename... Args>
      requires(sizeof...(Args) == dim)
    KOKKOS_FORCEINLINE_FUNCTION void operator()(Args &&...args) const
    {
      fun({{std::forward<Args>(args)...}});
    }

    FUN fun;
  };

  /**
   * @brief This is a functor which wraps a lambda for reduction.
   * Basically, this is necessary when one wants to call a variadic lambda on an NVIDIA GPU.
   * CUDA seems to be unable to expand the variadic arguments - in contrast, a direct approach does indeed work for
   * openMP or serial compilation.
   * To get around this limitation, the KokkosNDLambdaWrapperReduction packs the indices into an array.
   * Uses compile-time index sequences to extract the first `dim` args as indices and the last arg as the reduction
   * value, avoiding recursive tuple_first/tuple_cat overhead per GPU thread.
   *
   * @tparam dim Number of arguments taken
   * @tparam FUN The lambda to which we forward the indices
   */
  template <int dim, typename FUN> struct KokkosNDLambdaWrapperReduction {
    KOKKOS_FUNCTION
    KokkosNDLambdaWrapperReduction(const FUN &_fun) : fun(_fun) {};

    template <typename... Args>
      requires(sizeof...(Args) == dim + 1)
    KOKKOS_FORCEINLINE_FUNCTION void operator()(Args &&...args) const
    {
      impl(device::make_integer_sequence<size_t, dim>{}, device::forward<Args>(args)...);
    }

    FUN fun;

  private:
    template <size_t... Is, typename... Args>
    KOKKOS_FORCEINLINE_FUNCTION void impl(device::integer_sequence<size_t, Is...>, Args &&...args) const
    {
      auto tuple = device::tie(args...);
      fun(device::array<size_t, dim>{{static_cast<size_t>(device::get<Is>(tuple))...}}, device::get<dim>(tuple));
    }
  };
} // namespace DiFfRG

#include <autodiff/forward/real.hpp>

namespace Kokkos
{ // reduction identity must be defined in Kokkos namespace
  template <size_t N, class T> struct reduction_identity<autodiff::Real<N, T>> {
    KOKKOS_FORCEINLINE_FUNCTION static autodiff::Real<N, T> sum() { return autodiff::Real<N, T>(); }
  };
} // namespace Kokkos
