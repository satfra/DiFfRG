#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/kt_models.hh>
#include <boilerplate/timestepping.hh>

using namespace DiFfRG;

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test implicit euler with DG constant model", "[timestepping][constant][implicit_euler][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  // Serial on purpose: this tests a serial-only stepper, and a plain RectangularMesh is partitioned in an MPI build.
  using Discretization = DG::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperImplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>(
      "test_implicit_euler_constant_dg", 5e-2, "jacobian_diagnostics.csv",
      static_cast<double>(ImplicitTimestepperKind::implicit_euler), {0.}));
}
TEST_CASE("Jacobian diagnostics are off unless switched on", "[timestepping][implicit_euler][jacobian_diagnostics]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperImplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>(
      "test_implicit_euler_no_jacobian_diagnostics", 5e-2, "jacobian_diagnostics.csv",
      std::numeric_limits<double>::quiet_NaN(), {}, /*jacobian_diagnostics = */ false));
}
TEST_CASE("Test implicit euler with DG exponential model", "[timestepping][exponential][explicit_euler][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelExp<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperImplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_implicit_euler_exponential_dg", 5e-2));
}
TEST_CASE("Test implicit euler with DG Burgers model", "[timestepping][Burgers][explicit_euler][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperImplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_implicit_euler_burgers_dg", 5e-2));
}
TEST_CASE("Test implicit euler with KT Burgers model", "[timestepping][Burgers][implicit_euler][kt]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgersKT<dim>;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperImplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_implicit_euler_burgers_kt", 5e-2));
}
TEST_CASE("Test implicit euler with 2D KT Burgers model", "[timestepping][Burgers][implicit_euler][kt][2d]")
{
  constexpr uint dim = 2;
  using Model = Testing::ModelBurgers2DKT;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperImplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_implicit_euler_burgers_2d_kt", 6e-2));
}
TEST_CASE("Test implicit euler with KT viscous Burgers traveling wave",
          "[timestepping][Burgers][implicit_euler][kt][traveling_wave]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgersTravelingWaveKT<dim>;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperImplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_implicit_euler_traveling_wave_kt", 5e-2));
}
TEST_CASE("Test implicit euler with KT two-component Burgers system",
          "[timestepping][Burgers][implicit_euler][kt][two_component]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelTwoComponentBurgersKT<dim>;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperImplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_implicit_euler_two_component_burgers_kt", 5e-2));
}
