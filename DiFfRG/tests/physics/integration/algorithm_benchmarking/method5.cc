// Method 5: Host loop + two-phase per output point: MDRangePolicy parallel_for into 4D cache,
// then MDRangePolicy parallel_reduce from cache. Constant added in a separate final kernel.

#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration.hh>
#include <DiFfRG/physics/interpolation.hh>
#include <DiFfRG/physics/regulators.hh>

// #include "./ZA4/kernel.hh"
#include "./flows/ZA4/kernel.hh"

void method5(Catch::Benchmark::Chronometer &meter)
{
  // -------------------------------------------------------------------------------
  // Init: should be same across all methods.
  // -------------------------------------------------------------------------------

  using namespace DiFfRG;
  using NT = double;
  using ctype = double;
  using Regulator = DiFfRG::PolynomialExpRegulator<>;
  using KERNEL = DiFfRG::internal::Transform_p2_4D_3ang<NT, ZA4_kernel<Regulator>>;
  using ExecutionSpace = GPU_exec;

  static constexpr uint dim = 4;
  device::array<uint, dim> grid_size = {32, 8, 8, 8};
  device::array<double, dim> grid_min = {0.0, 0.0, -1.0, 0.0};
  device::array<double, dim> grid_max = {1.0, 1.0, 1.0, 2.0 * M_PI};
  device::array<DiFfRG::QuadratureType, dim> quadrature_type = {
      DiFfRG::QuadratureType::legendre, DiFfRG::QuadratureType::chebyshev2, DiFfRG::QuadratureType::legendre,
      DiFfRG::QuadratureType::legendre};

  device::array<ctype, dim> start;
  device::array<ctype, dim> scale;
  for (uint i = 0; i < dim; ++i) {
    start[i] = grid_min[i];
    scale[i] = (grid_max[i] - grid_min[i]);
  }

  device::array<Kokkos::View<const ctype *, GPU_memory>, dim> n;
  device::array<Kokkos::View<const ctype *, GPU_memory>, dim> w;

  QuadratureProvider quadrature_provider;

  for (uint i = 0; i < dim; ++i) {
    n[i] = quadrature_provider.template nodes<ctype, GPU_memory>(grid_size[i], quadrature_type[i]);
    w[i] = quadrature_provider.template weights<ctype, GPU_memory>(grid_size[i], quadrature_type[i]);
  }

  const LogCoordinates coordinates(64, 1e-3, 40., 10.);

  Kokkos::View<NT *, GPU_memory> integral_view("integral_view", coordinates.size());
  auto integral_view_host = Kokkos::create_mirror_view(integral_view);

  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> ZA3(coordinates);
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> ZA4(coordinates);
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> ZAcbc(coordinates);
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> dtZc(coordinates);
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> Zc(coordinates);
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> dtZA(coordinates);
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> ZA(coordinates);

  std::vector<double> dummy_data(coordinates.size(), 1.0);
  ZA3.update(dummy_data.data());
  ZA4.update(dummy_data.data());
  ZAcbc.update(dummy_data.data());
  dtZc.update(dummy_data.data());
  Zc.update(dummy_data.data());
  dtZA.update(dummy_data.data());
  ZA.update(dummy_data.data());

  const double k = 10.; // example value for k
  const auto m_args = device::make_tuple(k, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA);

  // -------------------------------------------------------------------------------
  // Method
  // -------------------------------------------------------------------------------

  // Kokkos
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamType = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  using Scratch = typename ExecutionSpace::scratch_memory_space;

  Kokkos::View<NT ****, GPU_memory> cache("cache", n[0].size(), n[1].size(), n[2].size(), n[3].size());

  meter.measure([&] {
    GPU_exec space;

    for (uint k = 0; k < integral_view.size(); ++k) {
      // get the position for the current index
      const auto idx = coordinates.from_linear_index(k);
      const auto pos = coordinates.forward(idx);
      // make a tuple of all arguments
      const auto full_args = device::tuple_cat(pos, m_args);

      auto subview = Kokkos::subview(integral_view, k);
      Kokkos::parallel_for(
          Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
              space, {0, 0, 0, 0}, {n[0].size(), n[1].size(), n[2].size(), n[3].size()}), // range of the kernel
          KOKKOS_LAMBDA(const uint idx0, const uint idx1, const uint idx2, const uint idx3) {
            const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
            const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
            const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
            const ctype x3 = Kokkos::fma(scale[3], n[3][idx3], start[3]);

            const ctype weight =
                w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2] * w[3][idx3] * scale[3];
            const NT result = device::apply(
                [&](const auto &...iargs) { return KERNEL::kernel(x0, x1, x2, x3, iargs...); }, full_args);
            cache(idx0, idx1, idx2, idx3) = weight * result;
          });

      // reduction
      Kokkos::parallel_reduce(
          "reduction",
          Kokkos::MDRangePolicy<Kokkos::Rank<4>>(space, {0, 0, 0, 0},
                                                 {n[0].size(), n[1].size(), n[2].size(), n[3].size()}),
          KOKKOS_LAMBDA(const uint idx0, const uint idx1, const uint idx2, const uint idx3, NT &update) {
            update += cache(idx0, idx1, idx2, idx3);
          },
          subview);
    }
    Kokkos::parallel_for(
        "Add_constant", Kokkos::RangePolicy<ExecutionSpace>(space, 0, integral_view.size()),
        KOKKOS_LAMBDA(const uint i) {
          // get the position for the current index
          const auto idx = coordinates.from_linear_index(i);
          const auto pos = coordinates.forward(idx);
          // make a tuple of all arguments
          const auto full_args = device::tuple_cat(pos, m_args);
          integral_view(i) +=
              device::apply([&](const auto &...iargs) { return KERNEL::constant(iargs...); }, full_args);
        });
    space.fence();
  });
  Kokkos::deep_copy(integral_view_host, integral_view);
  REQUIRE(is_close(integral_view_host[0] - 7.54411e-06, 0., 1e-11));
}
