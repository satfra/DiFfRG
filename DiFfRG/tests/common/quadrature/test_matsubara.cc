#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>

#include <DiFfRG/common/quadrature/matsubara.hh>

using namespace DiFfRG;

TEST_CASE("Test matsubara quadrature rule", "[double][quadrature][matsubara]")
{
  MatsubaraQuadrature<double> mq(1.0, 1.0);
  std::cout << "nodes: (";
  for (uint i = 0; i < mq.size(); ++i) {
    std::cout << mq.nodes()[i] << ", ";
  }
  std::cout << ")\n";
  std::cout << "weights: (";
  for (uint i = 0; i < mq.size(); ++i) {
    std::cout << mq.weights()[i] << ", ";
  }
  std::cout << ")\n";
}