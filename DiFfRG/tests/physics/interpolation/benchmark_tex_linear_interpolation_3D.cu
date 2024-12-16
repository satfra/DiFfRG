#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/physics/interpolation/linear_interpolation_3D.hh>
#include <DiFfRG/physics/interpolation/tex_linear_interpolation_3D.hh>

template <typename NT, typename LIN> __global__ void interp_kernel(NT *dest, LIN lin, float at1, float at2, float at3)
{
  uint idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto res = lin(at1, at2, at3);
  res = lin(at1 * 0.5, at2, at3);
  res = lin(at1, at2 * 0.5, at3);
  res = lin(at1, at2, at3 * 0.5);
  res = lin(at1, at2 * 0.5, at3);
  res = lin(at1 * 0.5, at2, at3);
  res = lin(at1, at2, at3);
  dest[idx_x] = res;
}

using namespace DiFfRG;

TEST_CASE("Test 3D gpu texture interpolation", "[1D][texture interpolation]")
{
  using Coordinates1D = LinearCoordinates1D<float>;
  using Coordinates3D = CoordinatePackND<Coordinates1D, Coordinates1D, Coordinates1D>;

  const float p1_start = GENERATE(take(1, random(1e-6, 1e-1)));
  const float p1_stop = GENERATE(take(1, random(1, 100))) + p1_start;
  const int p1_size = GENERATE(take(1, random(10, 100)));

  const float p2_start = GENERATE(take(1, random(1e-6, 1e-1)));
  const float p2_stop = GENERATE(take(1, random(1, 100))) + p2_start;
  const int p2_size = GENERATE(take(1, random(10, 100)));

  const float p3_start = GENERATE(take(1, random(1e-6, 1e-1)));
  const float p3_stop = GENERATE(take(1, random(1, 100))) + p3_start;
  const int p3_size = GENERATE(take(1, random(10, 100)));

  std::vector<float> in_data(p1_size * p2_size * p3_size, 0.);
  for (int i = 0; i < p1_size; ++i)
    for (int j = 0; j < p2_size; ++j)
      for (int k = 0; k < p3_size; ++k)
        in_data[i * p2_size * p3_size + j * p3_size + k] = k;

  Coordinates3D coords(Coordinates1D(p1_size, p1_start, p1_stop), Coordinates1D(p2_size, p2_start, p2_stop),
                       Coordinates1D(p3_size, p3_start, p3_stop));
  TexLinearInterpolator3D<float, Coordinates3D> interpolator(coords);
  interpolator.update(in_data.data());
  LinearInterpolator3D<float, Coordinates3D> lin_interpolator(coords);
  lin_interpolator.update(in_data.data());
  LinearInterpolator3D<double, Coordinates3D> double_lin_interpolator(coords);
  double_lin_interpolator.update(in_data.data());

  const uint n_el = GENERATE(1e+3, 1e+6, 1e+8);
  const float p1_pt = (p1_start + GENERATE(take(1, random(0., 1.))) * (p1_stop - p1_start));
  const float p2_pt = (p2_start + GENERATE(take(1, random(0., 1.))) * (p2_stop - p2_start));
  const float p3_pt = (p3_start + GENERATE(take(1, random(0., 1.))) * (p3_stop - p3_start));

  BENCHMARK_ADVANCED("TexInterpolator " + std::to_string(n_el))(Catch::Benchmark::Chronometer meter)
  {
    thrust::device_vector<float> dest(n_el, 0.);
    meter.measure([&] {
      interp_kernel<<<1, n_el>>>(thrust::raw_pointer_cast(dest.data()), interpolator, p1_pt, p2_pt, p3_pt);
      const auto res_device = thrust::reduce(dest.begin(), dest.end());
    });
  };
  BENCHMARK_ADVANCED("LinInterpolator " + std::to_string(n_el))(Catch::Benchmark::Chronometer meter)
  {
    thrust::device_vector<float> dest(n_el, 0.);
    meter.measure([&] {
      interp_kernel<<<1, n_el>>>(thrust::raw_pointer_cast(dest.data()), lin_interpolator, p1_pt, p2_pt, p3_pt);
      const auto res_device = thrust::reduce(dest.begin(), dest.end());
    });
  };
  BENCHMARK_ADVANCED("DoubleLinInterpolator " + std::to_string(n_el))(Catch::Benchmark::Chronometer meter)
  {
    thrust::device_vector<float> dest(n_el, 0.);
    meter.measure([&] {
      interp_kernel<<<1, n_el>>>(thrust::raw_pointer_cast(dest.data()), double_lin_interpolator, p1_pt, p2_pt, p3_pt);
      const auto res_device = thrust::reduce(dest.begin(), dest.end());
    });
  };
}