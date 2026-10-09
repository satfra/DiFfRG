/**
 * \mainpage DiFfRG C++ API
 *
 * \section mainpage_overview Overview
 *
 * **DiFfRG** (Discretization Framework for functional Renormalization Group flows) is a
 * C++20 scientific-computing library for solving fRG flow equations. It provides spatial
 * and temporal discretization of quantum field theory systems, built on
 * [deal.II](https://www.dealii.org/) (FEM), [Kokkos](https://kokkos.org/) (GPU/CPU
 * parallelization) and [SUNDIALS](https://computing.llnl.gov/projects/sundials) (implicit
 * solvers).
 *
 * This is the reference for the C++ library. A simulation is assembled from a small number
 * of components along a fixed pipeline:
 *
 * \section mainpage_pipeline The pipeline
 *
 * **Model → Discretization → Assembler → Timestepper → Output**
 *
 * -# **Entry point.** \ref DiFfRG::Init "Init" initializes the runtime (Kokkos, and MPI if
 *    enabled) and hands back a \ref DiFfRG::ConfigurationHelper "ConfigurationHelper" that
 *    parses the `parameter.json/toml` file and command-line overrides.
 * -# **Model.** The physics is defined by deriving from the CRTP base
 *    \ref DiFfRG::def::AbstractModel "def::AbstractModel", which fixes the mass, flux and
 *    source functions of the flow equations. Behaviour is composed from mixins:
 *    \ref DiFfRG::def::fRG "def::fRG" (RG time / cutoff scale),
 *    \ref DiFfRG::def::LLFFlux "def::LLFFlux" (local Lax-Friedrichs numerical flux),
 *    \ref DiFfRG::def::FlowBoundaries "def::FlowBoundaries" (inflow/outflow boundaries) and
 *    \ref DiFfRG::def::AD "def::AD" (automatic-differentiation Jacobians). Finite
 *    volume models additionally pick a boundary stencil, e.g.
 *    \ref DiFfRG::def::FVDefaultBoundaries "def::FVDefaultBoundaries" or
 *    \ref DiFfRG::def::RhoSymmetricLinearExtrapolationBoundaries
 *    "def::RhoSymmetricLinearExtrapolationBoundaries". The degrees of freedom are declared with
 *    \ref DiFfRG::ComponentDescriptor "ComponentDescriptor" /
 *    \ref DiFfRG::FEFunctionDescriptor "FEFunctionDescriptor" and \ref DiFfRG::Scalar "Scalar".
 * -# **Discretization.** The field space is discretized with one of
 *    \ref DiFfRG::CG::Discretization "CG", \ref DiFfRG::DG::Discretization "DG",
 *    \ref DiFfRG::LDG::Discretization "LDG" or \ref DiFfRG::FV::Discretization "FV"
 *    discretizations on a \ref DiFfRG::RectangularMesh "RectangularMesh" (whose grading and
 *    extents are configured through \ref DiFfRG::Config::ConfigurationMesh
 *    "Config::ConfigurationMesh").
 * -# **Assembler.** An assembler implementing \ref DiFfRG::AbstractAssembler
 *    "AbstractAssembler" computes residuals and Jacobians from the model and discretization;
 *    the concrete variants are \ref DiFfRG::CG::Assembler "CG", \ref DiFfRG::DG::Assembler
 *    "DG", \ref DiFfRG::LDG::Assembler "LDG" and
 *    \ref DiFfRG::FV::KurganovTadmor::Assembler "FV".
 * -# **Finite volume policies.** The Kurganov-Tadmor assembler takes its advective face
 *    reconstruction, its wave-speed estimate and (optionally) a separate reconstruction for the
 *    Jacobian as template policies. The reconstructor is either
 *    \ref DiFfRG::def::FirstOrderReconstructor "def::FirstOrderReconstructor" (piecewise
 *    constant) or \ref DiFfRG::def::TVDReconstructor "def::TVDReconstructor", which is in turn
 *    parameterized by a slope limiter — \ref DiFfRG::def::MinModLimiter "def::MinModLimiter"
 *    (the default), \ref DiFfRG::def::CentralLimiter "def::CentralLimiter",
 *    \ref DiFfRG::def::SuperbeeLimiter "def::SuperbeeLimiter" or
 *    \ref DiFfRG::def::VanAlbadaLimiter "def::VanAlbadaLimiter". The local wave speed entering
 *    the numerical flux comes from
 *    \ref DiFfRG::FV::KurganovTadmor::MaxEigenvalueWaveSpeed "MaxEigenvalueWaveSpeed"; diffusive
 *    fluxes always use
 *    \ref DiFfRG::def::CorrectedWeightedLeastSquaresDiffusionReconstructor
 *    "def::CorrectedWeightedLeastSquaresDiffusionReconstructor".
 * -# **Timestepper.** A timestepper implementing \ref DiFfRG::AbstractTimestepper
 *    "AbstractTimestepper" evolves the system in RG time. The recommended default is the
 *    differential-algebraic \ref DiFfRG::TimeStepperSUNDIALS_IDA "TimeStepperSUNDIALS_IDA";
 *    explicit (Euler, Runge-Kutta, Adams-Bashforth-Moulton) and further implicit
 *    (\ref DiFfRG::TimeStepperImplicitEuler "implicit Euler",
 *    \ref DiFfRG::TimeStepperTRBDF2 "TRBDF2") steppers are also provided. The solution is
 *    held in \ref DiFfRG::AbstractFlowingVariables "AbstractFlowingVariables"
 *    (\ref DiFfRG::FE::FlowingVariables "FE::FlowingVariables"). Implicit steppers solve their
 *    linear systems through an \ref DiFfRG::AbstractLinearSolver "AbstractLinearSolver", e.g.
 *    \ref DiFfRG::GMRES "GMRES", \ref DiFfRG::ScaledLinearSolver "ScaledLinearSolver" or
 *    \ref DiFfRG::UMFPack "UMFPack".
 * -# **Output.** Results are written through run-owned \ref DiFfRG::OutputSession
 *    "OutputSession" instances and scoped \ref DiFfRG::OutputFrame "OutputFrame" events
 *    (CSV / HDF5 / VTK) below an \ref DiFfRG::OutputPath "OutputPath"; diagnostic messages go
 *    to the \ref DiFfRG::ReportPort "ReportPort" the session hands out. The mesh can be refined
 *    adaptively with \ref DiFfRG::HAdaptivity "HAdaptivity".
 *
 * \section mainpage_integration Momentum-space integration
 *
 * Fully momentum-dependent flow equations are evaluated by integrators implementing
 * \ref DiFfRG::AbstractIntegrator "AbstractIntegrator", such as
 * \ref DiFfRG::Integrator_p2 "Integrator_p2", which share quadrature rules through a
 * \ref DiFfRG::QuadratureProvider "QuadratureProvider". The corresponding kernels are
 * generated from Mathematica (see the Mathematica reference and Tutorial 3 of the wider
 * documentation site).
 *
 * \section mainpage_next Getting started
 *
 * For installation instructions and worked, end-to-end examples, see the *Getting Started*
 * section and the *Tutorials* of the documentation site. Including the umbrella header
 * `#include <DiFfRG/DiFfRG.hh>` pulls in the whole library.
 */

#pragma once

// Common utilities
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/csv_reader.hh>
#include <DiFfRG/common/external_data_interpolator.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/interpolation.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/minimization.hh>
#include <DiFfRG/common/root_finding.hh>
#include <DiFfRG/common/run_reporter.hh>
#include <DiFfRG/common/utils.hh>

// Model
#include <DiFfRG/model/model.hh>

// Discretization
#include <DiFfRG/discretization/discretization.hh>

// Timestepping
#include <DiFfRG/timestepping/timestepping.hh>

// Physics
#include <DiFfRG/physics/integration.hh>
#include <DiFfRG/physics/interpolation.hh>
#include <DiFfRG/physics/physics.hh>
