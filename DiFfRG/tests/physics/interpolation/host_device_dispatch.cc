#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/kokkos.hh>

using namespace DiFfRG;

namespace
{
  /**
   * @brief The storage/dispatch core of the space-agnostic interpolators, in isolation.
   *
   * One device-resident allocation plus its host mirror, and an accessor that picks between them
   * with the KOKKOS_IF_ON_* macros. Those are a plain preprocessor selection under nvcc + a GNU
   * host compiler (Kokkos_Macros.hpp; the NV_IF_TARGET spelling is gated on KOKKOS_COMPILER_NVHPC),
   * so each compilation pass sees exactly one body and exactly one return.
   */
  struct Probe {
    using ViewType = Kokkos::View<double *, GPU_memory, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::host_mirror_type;

    Probe() : device_data("probe", 1), host_data(Kokkos::create_mirror_view(device_data)) {}

    // Shallow copy of BOTH views, valid in host and in device code -- the property the whole
    // design rests on. Carrying the HostSpace mirror into a CUDA closure is what Kokkos::DualView
    // does; the view tracker force-disables reference counting on the device side, so the host
    // pointer is copied and never dereferenced there.
    KOKKOS_DEFAULTED_FUNCTION Probe(const Probe &) = default;

    KOKKOS_FORCEINLINE_FUNCTION double value(const size_t i) const
    {
      KOKKOS_IF_ON_DEVICE((return device_data(i);))
      KOKKOS_IF_ON_HOST((return host_data(i);))
    }

    ViewType device_data;
    HostViewType host_data;
  };
} // namespace

TEST_CASE("Host/device dispatch picks the right buffer", "[interpolator]")
{
  DiFfRG::Init();

  Probe p;

  // Deliberately desynchronise the two buffers, so neither check can pass by accident.
  p.host_data(0) = 1.0;
  Kokkos::deep_copy(p.device_data, p.host_data); // device holds 1.0
  p.host_data(0) = 2.0;                          // host holds 2.0

  // (a) A host-side call reads the host mirror.
  CHECK(p.value(0) == 2.0);

  // (b) The whole object -- including the HostSpace mirror -- is captured by value into a kernel,
  // and the device side reads the device buffer. If the branches were inverted this would not
  // merely return the wrong number: View::operator() runs runtime_check_memory_access_violation,
  // which Kokkos::abort()s with "attempt to access inaccessible memory space".
  double res_device = 0.;
  Kokkos::parallel_reduce(
      "probe", Kokkos::RangePolicy<GPU_exec>(0, 1),
      KOKKOS_LAMBDA(const int, double &sum) { sum += p.value(0); }, res_device);

  // In a build without a separate device space create_mirror_view aliases the source, so there is
  // only one allocation and both sides necessarily agree.
  if constexpr (std::is_same_v<Probe::ViewType::memory_space, Probe::HostViewType::memory_space>)
    CHECK(res_device == 2.0);
  else
    CHECK(res_device == 1.0);

  // (c) A copy loses neither side.
  Probe q = p;
  CHECK(q.value(0) == 2.0);
}
