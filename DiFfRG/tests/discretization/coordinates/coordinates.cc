#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/discretization/coordinates/combined_coordinates.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test Coordinate template constraints", "[coordinates]")
{
  using namespace DiFfRG;

  // Test static assertions for coordinate types
  STATIC_REQUIRE(is_coordinates<LinCoordinates>);
  STATIC_REQUIRE(is_coordinates<LogCoordinates>);
  STATIC_REQUIRE(is_coordinates<LogLogCoordinates>);
  STATIC_REQUIRE(is_coordinates<LogLinCoordinates>);
  STATIC_REQUIRE(is_coordinates<LinLinCoordinates>);
  STATIC_REQUIRE(is_coordinates<LogLogLinCoordinates>);
  STATIC_REQUIRE(is_coordinates<LinLinLinCoordinates>);
  STATIC_REQUIRE(is_coordinates<LinPeriodicCoordinates>);
  STATIC_REQUIRE(is_coordinates<LogLinPeriodicCoordinates>);
  STATIC_REQUIRE(is_coordinates<LogLinLinPeriodicCoordinates>);
  STATIC_REQUIRE(!is_coordinates<int>);
  STATIC_REQUIRE(!is_coordinates<std::array<double, 3>>);

  // focused logarithmic coordinates and the packs built from them
  STATIC_REQUIRE(is_coordinates<FocusedLogCoordinates>);
  STATIC_REQUIRE(is_coordinates<FocusedLogCoordinates1D<float>>);
  STATIC_REQUIRE(is_coordinates<FocusedLogLinCoordinates>);
  STATIC_REQUIRE(is_coordinates<FocusedLogLinLinCoordinates>);
  STATIC_REQUIRE(is_coordinates<FocusedLogLinPeriodicCoordinates>);
  STATIC_REQUIRE(is_coordinates<FocusedLogLinLinPeriodicCoordinates>);
  STATIC_REQUIRE(is_coordinates<FocusedBosonicCoordinates1DFiniteT>);
  STATIC_REQUIRE(is_coordinates<FocusedFermionicCoordinates1DFiniteT>);

  // the radial axis of the finite-T coordinates is a template parameter which defaults to the
  // logarithmic one, so the historical two-argument spellings must keep working
  STATIC_REQUIRE(is_coordinates<BosonicCoordinates1DFiniteT<>>);
  STATIC_REQUIRE(is_coordinates<BosonicCoordinates1DFiniteT<int, float>>);
  STATIC_REQUIRE(is_coordinates<FermionicCoordinates1DFiniteT<int, double>>);
  STATIC_REQUIRE(std::is_same_v<BosonicCoordinates1DFiniteT<int, float>::radial_type, LogarithmicCoordinates1D<float>>);
  STATIC_REQUIRE(
      std::is_same_v<FocusedBosonicCoordinates1DFiniteT::radial_type, FocusedLogCoordinates1D<double>>);
}

TEST_CASE("Test coordinate periodicity traits", "[coordinates]")
{
  using namespace DiFfRG;

  STATIC_REQUIRE(is_periodic_coordinate_v<LinPeriodicCoordinates>);
  STATIC_REQUIRE(is_periodic_coordinate_v<LinearPeriodicCoordinates1D<float>>);
  STATIC_REQUIRE(!is_periodic_coordinate_v<LinCoordinates>);
  STATIC_REQUIRE(!is_periodic_coordinate_v<LogCoordinates>);
  STATIC_REQUIRE(!is_periodic_coordinate_v<FocusedLogCoordinates>);
  STATIC_REQUIRE(!is_periodic_coordinate_v<int>);

  // per-axis queries on packs
  STATIC_REQUIRE(is_periodic_axis_v<FocusedLogLinLinPeriodicCoordinates, 2>);
  STATIC_REQUIRE(!is_periodic_axis_v<FocusedLogLinLinPeriodicCoordinates, 1>);
  STATIC_REQUIRE(!is_periodic_axis_v<FocusedLogLinLinPeriodicCoordinates, 0>);
  STATIC_REQUIRE(is_periodic_axis_v<LogLinLinPeriodicCoordinates, 2>);
  STATIC_REQUIRE(!is_periodic_axis_v<LogLinLinPeriodicCoordinates, 1>);
  STATIC_REQUIRE(!is_periodic_axis_v<LogLinLinPeriodicCoordinates, 0>);
  STATIC_REQUIRE(is_periodic_axis_v<LogLinPeriodicCoordinates, 1>);
  STATIC_REQUIRE(!is_periodic_axis_v<LogLinLinCoordinates, 2>);

  // coordinate systems which do not expose their axes as types are never reported periodic
  STATIC_REQUIRE(!is_periodic_axis_v<LinCoordinates, 0>);
}
TEST_CASE("Test finite-T coordinates extend the Matsubara stack O(4)-symmetrically", "[coordinates]")
{
  using namespace DiFfRG;
  const double T = 0.01, p = 0.3;
  const LogarithmicCoordinates1D<double> radial(24, 1e-2, 6., 5.);

  // inside the stack: row of the nearest frequency, radial momentum unchanged
  const BosonicCoordinates1DFiniteT<int, double> bos(radial, 0, 4, T);
  const auto [b_in, bp_in] = bos.backward(2. * 2. * M_PI * T, p);
  CHECK(b_in == Catch::Approx(2.).epsilon(1e-14)); // a continuous row coordinate, not an integer
  CHECK(bp_in == Catch::Approx(radial.backward(p)));

  // beyond the last row (n = 3): read row 3 at the spatial momentum that keeps m^2 + p^2 fixed
  const double m = 7. * 2. * M_PI * T, m3 = 3. * 2. * M_PI * T;
  const auto [b_out, bp_out] = bos.backward(m, p);
  CHECK(b_out == 3);
  CHECK(bp_out == Catch::Approx(radial.backward(std::sqrt(p * p + m * m - m3 * m3))));

  // fermionic stack n = -4..3, (2n+1) pi T: both edges
  const FermionicCoordinates1DFiniteT<int, double> ferm(radial, -4, 4, T);
  const double mf = 11. * M_PI * T, mf_edge = 7. * M_PI * T;
  const auto [f_hi, fp_hi] = ferm.backward(mf, p);
  CHECK(f_hi == 7);
  CHECK(fp_hi == Catch::Approx(radial.backward(std::sqrt(p * p + mf * mf - mf_edge * mf_edge))));
  const auto [f_lo, fp_lo] = ferm.backward(-mf, p);
  CHECK(f_lo == 0);
  CHECK(fp_lo == Catch::Approx(radial.backward(std::sqrt(p * p + mf * mf - mf_edge * mf_edge))));
}
