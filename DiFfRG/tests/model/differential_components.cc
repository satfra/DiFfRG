#include <catch2/catch_all.hpp>

#include <DiFfRG/model/model.hh>

#include <cmath>

using namespace dealii;
using namespace DiFfRG;

namespace
{
  using Components = ComponentDescriptor<FEFunctionDescriptor<Scalar<"a">, Scalar<"b">, Scalar<"c">>>;

  struct DefaultMass : public def::AbstractModel<DefaultMass, Components> {
  };

  /// a: coefficient (u_a - 1) vanishes at u = 1; b: algebraic; c: enters only m_a.
  struct MixedMass : public def::AbstractModel<MixedMass, Components> {
    template <int dim, typename NT, typename Vector, typename Vector_dot, size_t n>
    void mass(std::array<NT, n> &m, const Point<dim> &x, const Vector &u, const Vector_dot &dt_u) const
    {
      m[0] = (u[0] - 1.) * dt_u[0] + dt_u[2];
      m[1] = u[1] - x[0];
      m[2] = u[2];
    }
  };

  /// Defined only for x > 2: no sample point is valid.
  struct UndefinedMass : public def::AbstractModel<UndefinedMass, Components> {
    template <int dim, typename NT, typename Vector, typename Vector_dot, size_t n>
    void mass(std::array<NT, n> &m, const Point<dim> &x, const Vector &, const Vector_dot &dt_u) const
    {
      for (uint i = 0; i < n; ++i)
        m[i] = std::log(x[0] - 3.) * dt_u[i];
    }
  };
} // namespace

TEST_CASE("differential_components probes mass()", "[model]")
{
  CHECK(DefaultMass().differential_components<2>() == std::vector<bool>{true, true, true});
  CHECK(MixedMass().differential_components<1>() == std::vector<bool>{true, false, true});
  CHECK_THROWS_AS(UndefinedMass().differential_components<1>(), std::runtime_error);
}
