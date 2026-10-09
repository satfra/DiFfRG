#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/kt_models.hh>
#include <boilerplate/timestepping.hh>
#include <DiFfRG/common/init.hh>

using namespace DiFfRG;

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test SUNDIALS IDA with DG constant model", "[timestepping][constant][sundials_ida][dg]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<Model, RectangularMesh<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_constant_dg", 1e-9,
                                                             "jacobian_diagnostics.csv",
                                                             static_cast<double>(ImplicitTimestepperKind::ida), {0.}));
}
TEST_CASE("Test SUNDIALS IDA with DG exponential model", "[timestepping][exponential][sundials_ida][dg]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelExp<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<Model, RectangularMesh<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_exponential_dg", 1e-9));
}
TEST_CASE("Test SUNDIALS IDA with DG Burgers model", "[timestepping][Burgers][sundials_ida][dg]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<Model, RectangularMesh<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_burgers_dg", 1e-9));
}
TEST_CASE("Test SUNDIALS IDA with CG constant model", "[timestepping][constant][sundials_ida][fem]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  using Discretization = CG::Discretization<Model, RectangularMesh<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = CG::Assembler<Discretization>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_constant_cg", 1e-9));
}
TEST_CASE("Test SUNDIALS IDA with CG exponential model", "[timestepping][exponential][sundials_ida][fem]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelExp<dim>;
  using NumberType = double;
  using Discretization = CG::Discretization<Model, RectangularMesh<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = CG::Assembler<Discretization>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_exponential_cg", 1e-9));
}
TEST_CASE("Test SUNDIALS IDA with CG Burgers model", "[timestepping][Burgers][sundials_ida][fem]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using NumberType = double;
  using Discretization = CG::Discretization<Model, RectangularMesh<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = CG::Assembler<Discretization>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_burgers_cg", 1e-9));
}

TEST_CASE("Test SUNDIALS IDA with KT Burgers model", "[timestepping][Burgers][sundials_ida][kt]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgersKT<dim>;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMesh<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_burgers_kt", 1e-9));
}

TEST_CASE("Test SUNDIALS IDA with 2D KT Burgers model", "[timestepping][Burgers][sundials_ida][kt][2d]")
{
  DiFfRG::Init();
  constexpr uint dim = 2;
  using Model = Testing::ModelBurgers2DKT;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMesh<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_burgers_2d_kt", 2e-3));
}

TEST_CASE("Test SUNDIALS IDA with KT viscous Burgers traveling wave",
          "[timestepping][Burgers][sundials_ida][kt][traveling_wave]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgersTravelingWaveKT<dim>;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMesh<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_traveling_wave_kt", 2e-5));
}

TEST_CASE("Test SUNDIALS IDA with KT two-component Burgers system",
          "[timestepping][Burgers][sundials_ida][kt][two_component]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelTwoComponentBurgersKT<dim>;
  using NumberType = double;
  using Discretization = FV::Discretization<Model, RectangularMesh<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_two_component_burgers_kt", 2e-5));
}
