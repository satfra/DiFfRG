#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/discretization/common/solution_sample.hh>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>

namespace
{
  using namespace dealii;
  using namespace DiFfRG;

  /// A 1D mesh with two blocks of different, uniform spacing: [0,0.5] in 10 cells and (0.5,1] in 4.
  void make_two_block_mesh(Triangulation<1> &tria)
  {
    std::vector<double> steps;
    for (uint i = 0; i < 10; ++i)
      steps.push_back(0.05);
    for (uint i = 0; i < 4; ++i)
      steps.push_back(0.125);
    GridGenerator::subdivided_hyper_rectangle(tria, {steps}, Point<1>(0.), Point<1>(1.));
  }
} // namespace

TEST_CASE("SolutionSample samples every cell, in order", "[discretization][solution-sample]")
{
  Triangulation<1> tria;
  make_two_block_mesh(tria);

  FE_Q<1> fe(2);
  DoFHandler<1> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  MappingQ1<1> mapping;

  // u(x) = 3x - 1, exact in this space, so values and gradients are checkable to round-off.
  class Linear : public Function<1>
  {
  public:
    double value(const Point<1> &x, const unsigned int) const override { return 3. * x[0] - 1.; }
  } linear;
  Vector<double> solution(dof_handler.n_dofs());
  VectorTools::interpolate(mapping, dof_handler, linear, solution);

  const auto sample = make_solution_sample(solution, dof_handler, mapping);

  REQUIRE(sample.size() == 14);
  REQUIRE(sample.n_components() == 1);

  for (size_t i = 0; i < sample.size(); ++i) {
    if (i > 0) CHECK(sample[i].point[0] > sample[i - 1].point[0]);
    CHECK(sample[i].values[0] == Catch::Approx(3. * sample[i].point[0] - 1.).margin(1e-12));
    CHECK(sample[i].gradients[0][0] == Catch::Approx(3.).margin(1e-12));
  }

  SECTION("cell_width reports the local spacing, including across a block boundary")
  {
    for (size_t i = 0; i < 10; ++i)
      CHECK(sample[i].cell_width == Catch::Approx(0.05).margin(1e-12));
    for (size_t i = 10; i < 14; ++i)
      CHECK(sample[i].cell_width == Catch::Approx(0.125).margin(1e-12));
  }

  SECTION("samples sit at cell centres")
  {
    CHECK(sample[0].point[0] == Catch::Approx(0.025).margin(1e-12));
    CHECK(sample[9].point[0] == Catch::Approx(0.475).margin(1e-12));
    CHECK(sample[10].point[0] == Catch::Approx(0.5625).margin(1e-12));
  }
}

TEST_CASE("The callback builder supplies what a DG0 space cannot", "[discretization][solution-sample]")
{
  Triangulation<1> tria;
  make_two_block_mesh(tria);

  FE_DGQ<1> fe(0);
  DoFHandler<1> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  MappingQ1<1> mapping;

  Vector<double> solution(dof_handler.n_dofs());
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    std::vector<types::global_dof_index> dofs(1);
    cell->get_dof_indices(dofs);
    solution[dofs[0]] = 3. * cell->center()[0] - 1.;
  }

  // The FE path is correct for the values but, the shape functions being constant, reports zero
  // gradients -- which is exactly why the finite-volume assembler reconstructs its own.
  const auto fe_sample = make_solution_sample(solution, dof_handler, mapping);
  REQUIRE(fe_sample.size() == 14);
  for (size_t i = 0; i < fe_sample.size(); ++i) {
    CHECK(fe_sample[i].values[0] == Catch::Approx(3. * fe_sample[i].point[0] - 1.).margin(1e-12));
    CHECK(fe_sample[i].gradients[0][0] == Catch::Approx(0.).margin(1e-12));
  }

  // The callback builder keeps the same geometry and ordering but lets the caller say what the
  // solution is -- here a hand-supplied exact gradient standing in for a reconstruction.
  const auto reconstructed = make_solution_sample<1, double>(
      dof_handler, mapping, 1u,
      [](const auto &, const Point<1> &x, std::vector<double> &values, std::vector<Tensor<1, 1, double>> &gradients) {
        values[0] = 3. * x[0] - 1.;
        gradients[0][0] = 3.;
      });

  REQUIRE(reconstructed.size() == fe_sample.size());
  for (size_t i = 0; i < reconstructed.size(); ++i) {
    CHECK(reconstructed[i].point[0] == Catch::Approx(fe_sample[i].point[0]).margin(1e-12));
    CHECK(reconstructed[i].cell_width == Catch::Approx(fe_sample[i].cell_width).margin(1e-12));
    CHECK(reconstructed[i].values[0] == Catch::Approx(fe_sample[i].values[0]).margin(1e-12));
    CHECK(reconstructed[i].gradients[0][0] == Catch::Approx(3.).margin(1e-12));
  }
}
