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

TEST_CASE("Test explicit euler with DG constant model", "[timestepping][constant][explicit_euler][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  // Serial on purpose: this tests a serial-only stepper, and a plain RectangularMesh is partitioned in an MPI build.
  using Discretization = DG::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperExplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper, false, true>("test_explicit_euler_constant_dg", 1e-4));
}
TEST_CASE("Test explicit euler with DG exponential model", "[timestepping][exponential][explicit_euler][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelExp<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperExplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper, false, true>("test_explicit_euler_exponential_dg", 1e-4));
}
TEST_CASE("Test explicit euler with DG Burgers model", "[timestepping][Burgers][explicit_euler][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperExplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper, false, true>("test_explicit_euler_burgers_dg", 1e-4));
}
TEST_CASE("Test explicit euler with KT Burgers model", "[timestepping][Burgers][explicit_euler][kt]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgersKT<dim>;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperExplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_explicit_euler_burgers_kt", 1e-4));
}
TEST_CASE("Test explicit euler with 2D KT Burgers model", "[timestepping][Burgers][explicit_euler][kt][2d]")
{
  constexpr uint dim = 2;
  using Model = Testing::ModelBurgers2DKT;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperExplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_explicit_euler_burgers_2d_kt", 2e-2));
}
TEST_CASE("Test explicit euler with KT viscous Burgers traveling wave",
          "[timestepping][Burgers][explicit_euler][kt][traveling_wave]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgersTravelingWaveKT<dim>;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperExplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_explicit_euler_traveling_wave_kt", 3e-4));
}
TEST_CASE("Test explicit euler with KT two-component Burgers system",
          "[timestepping][Burgers][explicit_euler][kt][two_component]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelTwoComponentBurgersKT<dim>;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperExplicitEuler<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_explicit_euler_two_component_burgers_kt", 3e-4));
}
