#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/models.hh>
#include <boilerplate/timestepping.hh>

using namespace DiFfRG;

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test TRBDF2 with DG constant model", "[timestepping][constant][trbdf2][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  // Serial on purpose: this tests a serial-only stepper, and a plain RectangularMesh is partitioned in an MPI build.
  using Discretization = DG::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperTRBDF2<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>(
      "test_trbdf2_constant_dg", 1e-3, "jacobian_diagnostics.csv", static_cast<double>(ImplicitTimestepperKind::trbdf2),
      {1., 2.}));
}
TEST_CASE("Test TRBDF2 with DG exponential model", "[timestepping][exponential][trbdf2][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelExp<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperTRBDF2<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_trbdf2_exponential_dg", 1e-3));
  // The space discretization is exact here, so this checks that the local error control limits the time error
  // (the global error accumulates over the steps, hence the per-step tolerance of precision / 100).
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_trbdf2_exponential_dg_tight", 1e-6));
}
TEST_CASE("Test TRBDF2 with DG Burgers model", "[timestepping][Burgers][trbdf2][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperTRBDF2<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_trbdf2_burgers_dg", 1e-3));
}
