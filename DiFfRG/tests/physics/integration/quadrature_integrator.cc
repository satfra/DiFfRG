#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "boilerplate/poly_integrand.hh"
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

TEMPLATE_TEST_CASE_SIG("Test ND quadrature integrals", "[integration][quadrature]", ((int dim), dim), (1), (2), (3),
                       (4), (5))
{
  DiFfRG::Init();

  auto check = [](auto execution_space, auto type) {
    using NT = std::decay_t<decltype(type)>;
    using ctype = typename get_type::ctype<NT>;
    using ExecutionSpace = std::decay_t<decltype(execution_space)>;

    using Kokkos::abs;
    auto t_abs = [](const auto val) {
      using type = std::decay_t<decltype(val)>;
      using Kokkos::abs;
      if constexpr (std::is_same_v<type, autodiff::real>)
        return abs(autodiff::val(val)) + abs(autodiff::grad(val));
      else if constexpr (std::is_same_v<type, cxreal>)
        return abs(autodiff::val(val)) + abs(autodiff::grad(val));
      else
        return abs(val);
    };

    std::array<ctype, dim> ext_min;
    std::array<ctype, dim> ext_max;
    std::array<size_t, dim> grid_size;
    std::array<QuadratureType, dim> quad_type;

    for (size_t d = 0; d < dim; ++d) {
      ext_min[d] = GENERATE(take(1, random(-2., -1.)));
      ext_max[d] = GENERATE(take(1, random(1., 2.)));
      grid_size[d] = d == 0 ? 32 : 16;
      quad_type[d] = QuadratureType::legendre;
    }

    QuadratureProvider quadrature_provider;
    QuadratureIntegrator<dim, NT, PolyIntegrand<dim, NT>, ExecutionSpace> integrator(quadrature_provider, grid_size,
                                                                                     ext_min, ext_max, quad_type);

    SECTION("Volume integral")
    {
      NT reference_integral = 1;
      for (uint d = 0; d < dim; ++d)
        reference_integral *= (ext_max[d] - ext_min[d]);

      std::array<NT, 4 * dim> coeffs{};
      for (uint d = 0; d < dim; ++d)
        coeffs[4 * d] = 1;

      NT integral{};
      std::apply([&](auto... coeffs) { integrator.get(integral, 0., coeffs...); }, coeffs);

      const ctype rel_err = t_abs(reference_integral - integral) / t_abs(reference_integral);
      if (rel_err >= expected_precision<ctype>::value) {
        std::cerr << "reference: " << reference_integral << "| integral: " << integral
                  << "| relative error: " << abs(reference_integral - integral) / abs(reference_integral) << std::endl;
      }
      CHECK(rel_err < expected_precision<ctype>::value);
    }
    SECTION("Volume map")
    {
      NT reference_integral = 1;
      for (uint d = 0; d < dim; ++d)
        reference_integral *= (ext_max[d] - ext_min[d]);

      std::array<NT, 4 * dim> coeffs{};
      for (uint d = 0; d < dim; ++d)
        coeffs[4 * d] = 1;

      const uint rsize = GENERATE(32, 64);
      std::vector<NT> integral_view(rsize);
      LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

      std::apply([&](auto... coeffs) { integrator.map(integral_view.data(), coordinates, coeffs...).fence(); }, coeffs);

      for (uint i = 0; i < rsize; ++i) {
        const ctype rel_err = t_abs(coordinates.forward(i) + reference_integral - integral_view[i]) /
                              t_abs(coordinates.forward(i) + reference_integral);
        if (rel_err >= 10 * expected_precision<ctype>::value) {
          std::cout << "reference: " << coordinates.forward(i) + reference_integral
                    << "| integral: " << integral_view[i] << "| relative error: "
                    << abs(coordinates.forward(i) + reference_integral - integral_view[i]) /
                           abs(coordinates.forward(i) + reference_integral)
                    << "| type: " << typeid(type).name() << "| i: " << i << std::endl;
        }
        CHECK(rel_err < 10 * expected_precision<ctype>::value);
      }
    };
    SECTION("Random polynomials")
    {
      std::vector<Polynomial> poly;
      for (uint d = 0; d < dim; ++d)
        poly.emplace_back(Polynomial({
            GENERATE(take(1, random(-1., 1.))), // x0
            GENERATE(take(1, random(-1., 1.))), // x1
            GENERATE(take(1, random(-1., 1.))), // x2
            GENERATE(take(1, random(-1., 1.))), // x3
        }));
      const NT constant = GENERATE(take(1, random(-1., 1.)));

      NT reference_integral = 1;
      for (uint d = 0; d < dim; ++d)
        reference_integral *= poly[d].integral(ext_min[d], ext_max[d]);
      reference_integral += constant;

      std::array<ctype, 4 * dim> coeffs{};
      for (uint d = 0; d < dim; ++d) {
        coeffs[4 * d] = poly[d][0];
        coeffs[4 * d + 1] = poly[d][1];
        coeffs[4 * d + 2] = poly[d][2];
        coeffs[4 * d + 3] = poly[d][3];
      }

      NT integral{};
      std::apply([&](auto... coeffs) { integrator.get(integral, constant, coeffs...); }, coeffs);

      const ctype rel_err = t_abs(reference_integral - integral) / t_abs(reference_integral);
      if (rel_err >= expected_precision<ctype>::value) {
        std::cerr << "reference: " << reference_integral << "| integral: " << integral << " | constant: " << constant
                  << " | relative error: " << abs(reference_integral - integral) / abs(reference_integral) << std::endl;
      }
      CHECK(rel_err < expected_precision<ctype>::value);
    }
  };

  SECTION("Threads integrals")
  {
    check(KokkosHost_exec(), (double)0);
    check(KokkosHost_exec(), (complex<double>)0);
    check(KokkosHost_exec(), (float)0);
    check(KokkosHost_exec(), (autodiff::real)0);
  };
  SECTION("TBB integrals")
  {
    check(TBB_exec(), (double)0);
    check(TBB_exec(), (complex<double>)0);
    check(TBB_exec(), (float)0);
    check(TBB_exec(), (autodiff::real)0);
  };
  SECTION("GPU integrals")
  {
    check(GPU_exec(), (double)0);
    check(GPU_exec(), (complex<double>)0);
    check(GPU_exec(), (float)0);
    check(GPU_exec(), (autodiff::real)0);
  };
}