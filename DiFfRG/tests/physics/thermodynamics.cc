#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/physics/thermodynamics.hh>
#include <DiFfRG/physics/threshold_functions/litim.hh>

#include <cmath>

using namespace DiFfRG;

namespace
{
  // Central finite difference of f at x.
  template <typename F> double fd(F f, const double x, const double h = 1e-5)
  {
    return (f(x + h) - f(x - h)) / (2. * h);
  }

  // Second central finite difference of f at x.
  template <typename F> double fd2(F f, const double x, const double h = 1e-4)
  {
    return (f(x + h) - 2. * f(x) + f(x - h)) / (h * h);
  }
} // namespace

TEST_CASE("Thermodynamic functions: derivatives and relations", "[physics][thermodynamics]")
{
  const double T = GENERATE(0.05, 0.37, 2.);
  const double e = GENERATE(0.3, 0.9, 1.7);
  INFO("T = " << T << ", e = " << e);

  CHECK(dcothS(e, T) == Catch::Approx(fd([&](double x) { return cothS(x, T); }, e)).epsilon(1e-6).margin(1e-9));
  CHECK(ddcothS(e, T) == Catch::Approx(fd([&](double x) { return dcothS(x, T); }, e)).epsilon(1e-6).margin(1e-9));
  CHECK(dddcothS(e, T) == Catch::Approx(fd([&](double x) { return ddcothS(x, T); }, e)).epsilon(1e-6).margin(1e-9));
  CHECK(dtanhS(e, T) == Catch::Approx(fd([&](double x) { return tanhS(x, T); }, e)).epsilon(1e-6).margin(1e-9));
  CHECK(ddtanhS(e, T) == Catch::Approx(fd([&](double x) { return dtanhS(x, T); }, e)).epsilon(1e-6).margin(1e-9));
  CHECK(dnB(e, T) == Catch::Approx(fd([&](double x) { return nB(x, T); }, e)).epsilon(1e-6).margin(1e-9));
  CHECK(ddnB(e, T) == Catch::Approx(fd([&](double x) { return dnB(x, T); }, e)).epsilon(1e-6).margin(1e-9));
  CHECK(dnF(e, T) == Catch::Approx(fd([&](double x) { return nF(x, T); }, e)).epsilon(1e-6).margin(1e-9));

  CHECK(cothS(e, T) == Catch::Approx(1. + 2. * nB(e, T)));
  CHECK(tanhS(e, T) == Catch::Approx(1. - 2. * nF(e, T)));
}

TEST_CASE("Litim threshold functions obey their mass recursion", "[physics][threshold_functions]")
{
  // B_{n+1} = -(1/n) dB_n/dm^2 and F_{n+1} = -(1/n) dF_n/dm^2.
  using namespace fRG::TFLitimSpatial;
  const double k = 1.;
  const double T = GENERATE(0.1, 0.3, 1.);
  const double mu = GENERATE(0., 0.2);
  const double m2 = GENERATE(-0.3, 0.4, 2.);
  INFO("T = " << T << ", mu = " << mu << ", m2 = " << m2);

  const auto B1 = [&](double x) { return B<1, double>(x, k, T); };
  const auto B3 = [&](double x) { return B<3, double>(x, k, T); };
  const auto F1 = [&](double x) { return F<1, double>(x, k, T, mu); };

  CHECK(B<2, double>(m2, k, T) == Catch::Approx(-fd(B1, m2)).epsilon(1e-6));
  CHECK(B<3, double>(m2, k, T) == Catch::Approx(fd2(B1, m2) / 2.).epsilon(1e-5));
  CHECK(B<4, double>(m2, k, T) == Catch::Approx(-fd(B3, m2) / 3.).epsilon(1e-6));
  CHECK(F<2, double>(m2, k, T, mu) == Catch::Approx(-fd(F1, m2)).epsilon(1e-6));
  CHECK(F<3, double>(m2, k, T, mu) == Catch::Approx(fd2(F1, m2) / 2.).epsilon(1e-5));
}
