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
