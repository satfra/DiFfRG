#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/model/component_descriptor.hh>

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test", "[model]")
{
  using namespace DiFfRG;

  using FEFunctions = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;

  constexpr FEFunctions idxf{};

  std::vector<int> data{GENERATE(take(10, random(0, 100))), GENERATE(take(10, random(0, 100))),
                        GENERATE(take(10, random(0, 100))), GENERATE(take(10, random(0, 100)))};

  [[maybe_unused]] constexpr auto idxu = idxf["u"];
  [[maybe_unused]] constexpr auto idxv = idxf["v"];

  const auto data_u = data[idxf["u"]];
  const auto data_v = data[idxf["v"]];

  REQUIRE(data_u == data[0]);
  REQUIRE(data_v == data[1]);
}

TEST_CASE("get_names_vector gives one name per component", "[model]")
{
  using namespace DiFfRG;

  using FEFunctions = FEFunctionDescriptor<Scalar<"u">, FunctionND<"phi", 3>, FunctionND<"A", 2, 2>, Scalar<"v">>;

  const std::vector<std::string> expected{"u", "phi_0", "phi_1", "phi_2", "A_0", "A_1", "A_2", "A_3", "v"};
  REQUIRE(FEFunctions::get_names_vector() == expected);
  REQUIRE(FEFunctions::get_names_vector().size() == FEFunctions::total_size);
}

TEST_CASE("LDG couplings and constant levels do not depend on the order of declaration", "[model]")
{
  using namespace DiFfRG;
  using Components = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>, VariableDescriptor<>,
                                         ExtractorDescriptor<>, FEFunctionDescriptor<Scalar<"du">, Scalar<"dv">>,
                                         FEFunctionDescriptor<Scalar<"ddu">>>;
  using Coupling = std::array<uint, 2>;

  Components top_down, bottom_up;
  top_down.add_dependency(2, 0, 1, 0);
  top_down.add_dependency(2, 0, 1, 1);
  top_down.add_dependency(1, 0, 0, 0);
  top_down.add_dependency(1, 1, 0, 1);
  top_down.add_dependency(1, 1, 0, 0);
  top_down.set_jacobian_constant(2, 1);
  top_down.set_jacobian_constant(1, 0);

  bottom_up.add_dependency(1, 0, 0, 0);
  bottom_up.add_dependency(1, 1, 0, 1);
  bottom_up.add_dependency(1, 1, 0, 0);
  bottom_up.add_dependency(2, 0, 1, 0);
  bottom_up.add_dependency(2, 0, 1, 1);
  bottom_up.set_jacobian_constant(1, 0);
  bottom_up.set_jacobian_constant(2, 1);

  for (const auto *c : {&top_down, &bottom_up}) {
    // ddu <- du <- u and ddu <- dv <- {u, v}, each pair once
    auto composed = c->ldg_couplings(2, 0);
    std::sort(composed.begin(), composed.end());
    CHECK(composed == std::vector<Coupling>{{0, 0}, {0, 1}});
    CHECK(c->jacobians_constant(2, 1));
    CHECK(c->jacobians_constant(2, 0));
  }

  Components partly_constant;
  partly_constant.set_jacobian_constant(2, 1);
  CHECK(partly_constant.jacobians_constant(2, 1));
  CHECK_FALSE(partly_constant.jacobians_constant(2, 0));
  CHECK_FALSE(partly_constant.jacobians_constant(1, 0));
  CHECK_THROWS(partly_constant.set_jacobian_constant(2, 0));
}
