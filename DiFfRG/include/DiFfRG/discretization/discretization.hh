#pragma once

// clang-format off
/**
 * @file discretization.hh
 * @brief Umbrella header of the spatial discretizations, and how to choose one.
 *
 * ### What is discretized
 * A model (see model/model.hh) describes \f$N\f$ FE functions \f$u_c(x, t)\f$ on a field-space domain
 * \f$x \in \Omega \subset \mathbb{R}^{d}\f$, evolved in RG time \f$t\f$ by
 * \f[
 *   m_c(u, \partial_t u) + \nabla \cdot F_c(u, \nabla u, \dots) + s_c(u, \nabla u, \dots) = 0\,, \qquad c = 1, \dots, N,
 * \f]
 * with the mass \f$m\f$ (Model::mass, default \f$m = \partial_t u\f$), the flux \f$F\f$ (Model::flux) and the
 * source \f$s\f$ (Model::source). Flux and source may also read the extractors \f$e\f$ (numbers taken from the
 * solution at one point, e.g. at the minimum of the potential) and the variables \f$v\f$ (extra scalars with their own
 * ODE, Model::dt_variables). A discretization turns this into the residual \f$m(U, \dot U) + R(U, v) = 0\f$ for the
 * vector \f$U\f$ of unknowns, which a timestepper integrates (see timestepping/abstract_timestepper.hh).
 *
 * Constrained dofs (hanging nodes of an adapted CG mesh, the model's affine constraints) are not unknowns: every
 * assembler applies the constraints to \f$U\f$ before evaluating, and these dofs are algebraic for IDA.
 *
 * ### The chain of types
 * A simulation is assembled from
 * ```
 * Model  ->  Discretization<Model, Mesh>  ->  Assembler<Discretization>  ->  TimeStepper<Assembler>
 * ```
 * The Discretization owns the FE (or FV) space on the mesh; the Assembler evaluates the model on it and provides
 * residuals and jacobians; the TimeStepper integrates. Minimal CG run (from Examples/ONfiniteT/CG.cc):
 * @code
 * using Model = ON_finiteT;
 * using Discretization = CG::Discretization<Model, RectangularMesh<Model::dim>>;
 * using Assembler = CG::Assembler<Discretization>;
 * using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;
 *
 * const auto config = DiFfRG::Init(argc, argv).get_configuration_helper().get_config();
 * Model model(config);
 * RectangularMesh<Model::dim> mesh{Config::ConfigurationMesh<Model::dim>(config)};
 * OutputSession<Assembler> data_out(config);
 * Discretization discretization(mesh, config, data_out.report_port());
 * Assembler assembler(discretization, model, config);
 * HAdaptivity mesh_adaptor(assembler, config);
 * TimeStepper time_stepper(config, assembler, data_out, mesh_adaptor);
 *
 * FE::FlowingVariables initial_condition(discretization);
 * initial_condition.interpolate(model);
 * time_stepper.run(initial_condition, 0., config.get_double("/timestepping/final_time"));
 * @endcode
 * The other schemes swap the two type aliases, e.g. `FV::Discretization` with `FV::KurganovTadmor::Assembler` (and
 * NoAdaptivity, FV::FlowingVariables). A flow without FE functions skips the discretization:
 * `Variables::Assembler<Model> assembler(model, config)`.
 *
 * ### Choosing a scheme
 * | Scheme       | Classes                                   | Unknowns                                  | Assembled                                                    | Further model hooks                                 | Mesh adaptation | MPI |
 * |--------------|-------------------------------------------|-------------------------------------------|--------------------------------------------------------------|-----------------------------------------------------|-----------------|-----|
 * | CG           | CG::Discretization, CG::Assembler         | nodal values of continuous \f$Q_p\f$ elements | weak form; flux enters only through the boundary             | boundary_numflux, cell_indicator                    | HAdaptivity     | yes |
 * | DG           | DG::Discretization, DG::Assembler         | discontinuous \f$Q_p\f$ elements           | CG terms plus a numerical flux on every interior face        | numflux (e.g. def::LLFFlux), face_indicator          | HAdaptivity     | yes |
 * | LDG          | LDG::Discretization, LDG::Assembler       | DG for \f$u\f$, plus 1 to 3 auxiliary levels  | DG for \f$u\f$; each level \f$L_k \approx \nabla\cdot F^{(k)} + s^{(k)}\f$ of the level below | ldg_flux<k>, ldg_source<k>, ldg_numflux, ldg_boundary_numflux | HAdaptivity | no |
 * | FV (KT)      | FV::Discretization, FV::KurganovTadmor::Assembler | cell averages                     | Kurganov-Tadmor central-upwind fluxes of a slope-limited reconstruction | diffusion_flux, apply_boundary_stencil, wave_speed_blocks | no (NoAdaptivity) | yes |
 * | Variables    | Variables::Assembler<Model>               | variables \f$v\f$ only                     | Model::dt_variables                                          | dt_variables                                        | -               | no  |
 *
 * - CG is the default for smooth solutions. DG and LDG allow jumps between cells, coupled through a numerical flux
 *   on the faces, at the cost of more unknowns; LDG gives the model derivatives of \f$u\f$ as independent FE fields
 *   (e.g. \f$\partial_x u\f$ in a non-linear flux) instead of the FE gradient.
 * - FV/KT is for convection-dominated flows that develop shocks (e.g. flows of the effective potential with
 *   convexity restoration). It needs a rectangular mesh and \f$d = 1\f$ or 2; with a TVD limiter it is second order
 *   in smooth regions and first order at extrema.
 * - All PDE schemes support extractors and variables. KT keeps them fixed within a Newton step (its jacobian has no
 *   extractor or variable dependence).
 * - MPI: CG, DG and KT run distributed with `RectangularMesh<dim>` (partitioned in an MPI build) or
 *   `RectangularMeshParallel<dim>`. LDG requires `RectangularMeshSerial<dim>`.
 *
 * ### Parameters
 * | Key                                          | Type | Default            | Used by | Meaning |
 * |----------------------------------------------|------|--------------------|---------|---------|
 * | `/discretization/fe_order`                   | uint | 3 (warns)          | CG, DG, LDG | polynomial degree \f$p\f$ (FV: ignored, warns if nonzero) |
 * | `/discretization/overintegration`            | uint | 0                  | all     | extra quadrature points: FEM uses QGauss(\f$p + 1 +\f$ oi) per direction; KT see FV::KurganovTadmor::Assembler |
 * | `/discretization/batched/max_stacked_points` | uint | budget / point size | all    | points per automatic-differentiation evaluation of the jacobian; default budget 256 MiB (FEM), 16 MiB (KT) |
 * | `/discretization/output_subdivisions`        | uint | 1                  | output  | subdivisions of each cell in VTK output |
 * | `/discretization/diagnose_flux_conditioning` | bool | false              | KT      | warn if a diffusion flux is dominated by a gradient-independent constant |
 *
 * Mesh adaptation (CG, DG, LDG) is controlled by `/discretization/adaptivity/{start_adapt_at, adapt_dt, level,
 * refine_percent, coarsen_percent, coarsen_buffer, coarsen_age}` (defaults 0, 0.1, 0 = off, 0.1, 0.05, 2, 2); see
 * HAdaptivity in mesh/h_adaptivity.hh. The keys of the extractor search (EoM, raw potential) are documented in
 * common/eom_config.hh.
 */
// clang-format on

#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FEM/dg.hh>
#include <DiFfRG/discretization/FEM/ldg.hh>

// Finite volume (Kurganov-Tadmor). The discretization pulls in the KT assembler; the
// reconstructors, limiters and wave speeds that parameterize it are listed explicitly so that
// every choice a user can make is available from the umbrella header.
#include <DiFfRG/discretization/FV/discretization.hh>

#include <DiFfRG/discretization/FV/limiter/central_limiter.hh>
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/limiter/superbee_limiter.hh>
#include <DiFfRG/discretization/FV/limiter/van_albada_limiter.hh>

#include <DiFfRG/discretization/FV/reconstructor/advection/first_order_reconstructor.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>
#include <DiFfRG/discretization/FV/reconstructor/diffusion/corrected_weighted_least_squares_reconstructor.hh>

#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh>

#include <DiFfRG/discretization/variables/variables.hh>

#include <DiFfRG/discretization/common/eom_config.hh>

#include <DiFfRG/discretization/data/csv_output.hh>
#include <DiFfRG/discretization/data/data.hh>
// DataOutput: a compile-time error that points to OutputSession.
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/discretization/data/output_path.hh>
#include <DiFfRG/discretization/data/output_settings.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/data/hdf5_input.hh>
#include <DiFfRG/discretization/data/hdf5_output.hh>

#include <DiFfRG/discretization/mesh/h_adaptivity.hh>
#include <DiFfRG/discretization/mesh/no_adaptivity.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>

#include <DiFfRG/discretization/coordinates/combined_coordinates.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/discretization/coordinates/stack_coordinates.hh>
