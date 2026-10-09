#pragma once

#include <DiFfRG/model/model.hh>

using namespace DiFfRG;

struct Parameters {
  Parameters(const ConfigTree &value)
  {
    try {
      a = value.get_double("/physical/a");
      b = value.get_double("/physical/b");
      c = value.get_double("/physical/c");
      d = value.get_double("/physical/d");
    } catch (std::exception &e) {
      std::cout << "Error in reading parameters: " << e.what() << std::endl;
    }
  }
  double a, b, c, d;
};

using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
using Components = ComponentDescriptor<FEFunctionDesc>;
constexpr auto idxf = FEFunctionDesc{};

/**
 * @brief This class implements the numerical model for Burgers' equation.
 */
class Tut1 : public def::AbstractModel<Tut1, Components>,
             public def::fRG,                  // this handles the fRG time
             public def::FlowBoundaries<Tut1>, // use Inflow/Outflow boundaries
             public def::AD<Tut1>              // define all jacobians per AD
{
private:
  const Parameters prm;

public:
  static constexpr uint dim = 1;

  Tut1(const ConfigTree &json) : def::fRG(json.get_double("/physical/Lambda")), prm(json) {}

  template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
  {
    const auto x = pos[0];
    values[idxf("u")] = prm.a + prm.b * powr<1>(x) + prm.c * powr<2>(x) + prm.d * powr<3>(x);
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux, const Point<dim> & /*pos*/,
            const Solution &sol) const
  {
    const auto &fe_functions = get<"fe_functions">(sol);

    const auto u = fe_functions[idxf("u")];

    flux[idxf("u")][0] = 0.5 * powr<2>(u);
  }
};
