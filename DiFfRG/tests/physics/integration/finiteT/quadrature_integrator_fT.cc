#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

#include "../boilerplate/poly_integrand.hh"

#include <cstring>

//--------------------------------------------
// Quadrature integration

TEMPLATE_TEST_CASE_SIG("Test finite temperature quadrature integrals", "[integration][quadrature]", ((int dim), dim),
                       (1), (2), (3), (4))
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

    constexpr int sdim = dim - 1; // spatial dimension

    std::array<ctype, sdim> ext_min;
    std::array<ctype, sdim> ext_max;
    std::array<size_t, sdim> grid_size;
    std::array<QuadratureType, sdim> quad_type;

    for (int d = 0; d < sdim; ++d) {
      ext_min[d] = GENERATE(take(1, random(-2., -1.)));
      ext_max[d] = GENERATE(take(1, random(1., 2.)));
      grid_size[d] = d == 0 ? 32 : 16;
      quad_type[d] = QuadratureType::legendre;
    }

    const ctype T = GENERATE(take(1, random(0.01, 1.)));

    QuadratureProvider quadrature_provider;
    QuadratureIntegrator_fT<dim, NT, PolyIntegrand<sdim + 1, NT, -1>, ExecutionSpace> integrator(
        quadrature_provider, grid_size, ext_min, ext_max, quad_type);
    integrator.set_T(T);

    SECTION("Volume integral")
    {
      const ctype val = GENERATE(take(1, random(0., 1.)));
      integrator.set_typical_E(val);

      NT reference_integral = 1;
      for (int d = 0; d < sdim; ++d)
        reference_integral *= (ext_max[d] - ext_min[d]);
      reference_integral /= std::tanh(val / (2. * T)) * 2. * val; // sum

      std::array<NT, 4 * (sdim + 1)> coeffs{};
      for (int d = 0; d < sdim; ++d)
        coeffs[4 * d] = 1;
      coeffs[4 * sdim] = powr<2>(val);
      coeffs[4 * sdim + 2] = powr<2>(1);

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
      const ctype val = GENERATE(take(1, random(0., 1.)));
      integrator.set_typical_E(val);

      NT reference_integral = 1;
      for (int d = 0; d < sdim; ++d)
        reference_integral *= (ext_max[d] - ext_min[d]);
      reference_integral /= std::tanh(val / (2. * T)) * 2. * val; // sum

      std::array<NT, 4 * (sdim + 1)> coeffs{};
      for (int d = 0; d < sdim; ++d)
        coeffs[4 * d] = 1;
      coeffs[4 * sdim] = powr<2>(val);
      coeffs[4 * sdim + 2] = powr<2>(1);

      const size_t rsize = GENERATE(32, 64);
      std::vector<NT> integral_view(rsize);
      LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

      std::apply([&](auto... coeffs) { integrator.map(integral_view.data(), coordinates, coeffs...).fence(); }, coeffs);

      for (size_t i = 0; i < rsize; ++i) {
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
  };

  // Check on TBB
  SECTION("TBB") { check(TBB_exec(), (double)0); }
  // Check on Threads
  SECTION("Threads") { check(KokkosHost_exec(), (double)0); }
  // Check on GPU
  SECTION("GPU") { check(GPU_exec(), (double)0); }
}
TEST_CASE("Finite-T device maps stage through pinned memory and defer", "[integration][quadrature]")
{
  DiFfRG::Init();

  // The finite-T integrator used to copy device -> `dest` directly. `dest` is ordinary pageable
  // caller memory, so that cudaMemcpyAsync is not actually asynchronous: the driver stages it and
  // the call blocks until the kernels feeding it have finished. The result was zero host run-ahead
  // -- the device drained after every single map(), which is exactly what MapCompletion was built
  // to avoid for the vacuum integrator. This pins the fix: results must be staged and deferred,
  // and must still be bitwise what the undeferred path produces.
  constexpr int dim = 2;
  constexpr int sdim = dim - 1;
  using NT = double;
  using ctype = get_type::ctype<NT>;

  auto check = [](auto execution_space) {
    using ExecutionSpace = std::decay_t<decltype(execution_space)>;

    std::array<ctype, sdim> ext_min{-1.5}, ext_max{1.25};
    std::array<size_t, sdim> grid_size{32};
    std::array<QuadratureType, sdim> quad_type{QuadratureType::legendre};

    QuadratureProvider qp;
    QuadratureIntegrator_fT<dim, NT, PolyIntegrand<sdim + 1, NT, -1>, ExecutionSpace> a(qp, grid_size, ext_min,
                                                                                        ext_max, quad_type);
    QuadratureIntegrator_fT<dim, NT, PolyIntegrand<sdim + 1, NT, -1>, ExecutionSpace> b(qp, grid_size, ext_min,
                                                                                        ext_max, quad_type);
    a.set_T(0.3);
    b.set_T(0.3);

    std::array<NT, 4 * (sdim + 1)> coeffs{};
    coeffs[0] = 1;
    coeffs[4 * sdim] = powr<2>(NT(1.7));
    coeffs[4 * sdim + 2] = 1;

    constexpr size_t rsize = 32;
    LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

    // Undeferred reference: map() lands before it returns, as it always has outside a scope.
    std::vector<NT> ref_a(rsize, NT(0)), ref_b(rsize, NT(0));
    std::apply([&](auto... c) { a.map(ref_a.data(), coordinates, c...); }, coeffs);
    std::apply([&](auto... c) { b.map(ref_b.data(), coordinates, c...); }, coeffs);

    std::vector<NT> got_a(rsize, NT(0)), got_b(rsize, NT(0));
    {
      DeferredMaps defer;
      std::apply([&](auto... c) { a.map(got_a.data(), coordinates, c...); }, coeffs);
      std::apply([&](auto... c) { b.map(got_b.data(), coordinates, c...); }, coeffs);

      // Held in pinned staging, not yet landed -- the host ran ahead instead of blocking on the
      // copy. On a host backend there is no staging and the result is already in place, by design.
      if constexpr (!std::is_same_v<typename ExecutionSpace::memory_space, CPU_memory>) {
        bool untouched = true;
        for (size_t i = 0; i < rsize; ++i)
          untouched &= got_a[i] == NT(0) && got_b[i] == NT(0);
        CHECK(untouched);
      }
    }

    CHECK(std::memcmp(ref_a.data(), got_a.data(), rsize * sizeof(NT)) == 0);
    CHECK(std::memcmp(ref_b.data(), got_b.data(), rsize * sizeof(NT)) == 0);

    // Sanity: not zeros that happen to match.
    bool nonzero = false;
    for (size_t i = 0; i < rsize; ++i)
      nonzero |= got_a[i] != NT(0);
    CHECK(nonzero);
  };

  SECTION("GPU") { check(GPU_exec()); }
  SECTION("Threads") { check(KokkosHost_exec()); }
  SECTION("TBB") { check(TBB_exec()); }
}

TEST_CASE("A second finite-T map from one integrator lands the first", "[integration][quadrature]")
{
  DiFfRG::Init();

  // One staging buffer per integrator, so mapping twice inside a scope must land the first result
  // before the second overwrites the buffer.
  constexpr int dim = 2;
  constexpr int sdim = dim - 1;
  using NT = double;
  using ctype = get_type::ctype<NT>;

  std::array<ctype, sdim> ext_min{-1.5}, ext_max{1.25};
  std::array<size_t, sdim> grid_size{32};
  std::array<QuadratureType, sdim> quad_type{QuadratureType::legendre};

  QuadratureProvider qp;
  QuadratureIntegrator_fT<dim, NT, PolyIntegrand<sdim + 1, NT, -1>, GPU_exec> integrator(qp, grid_size, ext_min,
                                                                                         ext_max, quad_type);
  integrator.set_T(0.3);

  std::array<NT, 4 * (sdim + 1)> coeffs{};
  coeffs[0] = 1;
  coeffs[4 * sdim] = powr<2>(NT(1.7));
  coeffs[4 * sdim + 2] = 1;

  constexpr size_t rsize = 32;
  LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

  std::vector<NT> reference(rsize, NT(0));
  std::apply([&](auto... c) { integrator.map(reference.data(), coordinates, c...); }, coeffs);

  std::vector<NT> first(rsize, NT(0)), second(rsize, NT(0));
  {
    DeferredMaps defer;
    std::apply([&](auto... c) { integrator.map(first.data(), coordinates, c...); }, coeffs);
    std::apply([&](auto... c) { integrator.map(second.data(), coordinates, c...); }, coeffs);
  }

  CHECK(std::memcmp(reference.data(), first.data(), rsize * sizeof(NT)) == 0);
  CHECK(std::memcmp(reference.data(), second.data(), rsize * sizeof(NT)) == 0);
}
