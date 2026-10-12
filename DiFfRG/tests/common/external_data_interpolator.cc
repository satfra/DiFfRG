#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <filesystem>
#include <fstream>

#include <DiFfRG/common/external_data_interpolator.hh>

using namespace DiFfRG;

TEST_CASE("ExternalDataInterpolator clamps to the data range for negative x", "[common][interpolation]")
{
  const auto file = std::filesystem::temp_directory_path() / "diffrg_edi_negative_x.csv";
  {
    std::ofstream out(file);
    // y = x^2 on x in [-3, -1]
    for (int i = 0; i <= 20; ++i) {
      const double x = -3. + 0.1 * i;
      out << x << "," << x * x << "\n";
    }
  }

  ExternalDataInterpolator interpolator({file.string()}, {}, ',', false, 0, 1);
  REQUIRE(interpolator.value(-2., 1) == Catch::Approx(4.).epsilon(1e-6));
  // Beyond the data the value at the nearest end is returned.
  REQUIRE(interpolator.value(-0.5, 1) == Catch::Approx(1.).epsilon(1e-6));
  REQUIRE(interpolator.value(-4., 1) == Catch::Approx(9.).epsilon(1e-6));

  std::filesystem::remove(file);
}
