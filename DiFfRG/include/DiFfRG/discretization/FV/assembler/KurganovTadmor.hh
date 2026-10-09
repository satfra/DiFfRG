#pragma once

// external libraries

// DiFfRG
#include "DiFfRG/common/math.hh"
#include <array>
#include <autodiff/forward/real/real.hpp>
#include <cstddef>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_iterator_base.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <spdlog/spdlog.h>
#include <sstream>
#include <tbb/tbb.h>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/first_order_reconstructor.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>
#include <DiFfRG/discretization/common/assembler_core.hh>
#include <DiFfRG/discretization/common/batched_scatter.hh>
#include <DiFfRG/discretization/common/eom.hh>
#include <DiFfRG/discretization/common/la_policy.hh>
#include <DiFfRG/discretization/common/phase_times.hh>
#include <DiFfRG/discretization/common/solution_sample.hh>
#include <DiFfRG/model/abs_tolerances.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>

#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/discretization/FV/assembler/assembly_context.hh>
#include <DiFfRG/discretization/FV/assembler/flux_jacobian_hessian.hh>
#include <DiFfRG/discretization/FV/assembler/flux_ties.hh>
#include <DiFfRG/discretization/FV/assembler/reconstruction_cache.hh>
#include <DiFfRG/discretization/FV/assembler/trace_batch.hh>
#include <DiFfRG/discretization/FV/wave_speed/abstract_wave_speed.hh>
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh>
#include <DiFfRG/discretization/common/types.hh>
#include <tuple>
#include <utility>
#include <vector>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      using namespace dealii;

      namespace internal
      {
        /// Per-thread workspace of the phase-3 passes: the cell's solution at its quadrature points and the
        /// reconstruction derivatives of the jacobian chain rule.
        template <int dim, typename NumberType, size_t n_components> struct ScratchData {
          using QuadratureValue = std::array<NumberType, n_components>;

          ScratchData(const dealii::Quadrature<dim> &quadrature)
              : solution_values(quadrature.size()), solution_dot_values(quadrature.size())
          {
          }

          std::vector<QuadratureValue> solution_values;
          std::vector<QuadratureValue> solution_dot_values;
          std::array<std::vector<ReconstructionDerivativeData<dim, NumberType, n_components>>, 2>
              reconstructed_derivatives;
          std::array<std::vector<ReconstructionDerivativeData<dim, NumberType, n_components>>, 2> diffusion_derivatives;
          // d(cell-centre gradient)/d(u_j) for the nonlocal part of the source jacobian. Separate from
          // reconstructed_derivatives, which face_jacobian() resizes for each face's dependency set.
          std::vector<GradientType<dim, NumberType, n_components>> source_gradient_derivatives;
          // d(cell-centre hessian)/d(u_j), only filled for models with source_uses_hessians.
          std::vector<std::array<dealii::Tensor<2, dim, NumberType>, n_components>> source_hessian_derivatives;
          CellStencilData<dim, NumberType, n_components> cell_stencil;
        };

        /**
         * @brief Copy of @p J with every row and column outside @p block zeroed.
         *
         * The spectral radius of the result is that of the block's own diagonal sub-matrix, padded
         * with zero eigenvalues, so a wave-speed strategy can be handed this and needs to know
         * nothing about blocks. @see AbstractModel::wave_speed_blocks.
         */
        template <typename NumberType, int dim, size_t n_components>
        std::array<JacobianMatrix<NumberType, n_components>, dim>
        restrict_jacobian_to_block(const std::array<JacobianMatrix<NumberType, n_components>, dim> &J,
                                   const std::array<int, n_components> &blocks, const int block)
        {
          std::array<JacobianMatrix<NumberType, n_components>, dim> restricted{};
          for (size_t d = 0; d < dim; ++d)
            for (size_t i = 0; i < n_components; ++i) {
              if (blocks[i] != block) continue;
              for (size_t j = 0; j < n_components; ++j)
                if (blocks[j] == block) restricted[d][i][j] = J[d][i][j];
            }
          return restricted;
        }

        /**
         * @brief Copy of @p H with the two jacobian indices restricted to @p block.
         *
         * The differentiation index is left alone: the block's speed depends on the whole solution
         * through the entries of its own sub-matrix. @see restrict_jacobian_to_block.
         */
        template <typename NumberType, int dim, size_t n_components>
        HessianTensor<NumberType, dim, n_components>
        restrict_hessian_to_block(const HessianTensor<NumberType, dim, n_components> &H,
                                  const std::array<int, n_components> &blocks, const int block)
        {
          HessianTensor<NumberType, dim, n_components> restricted{};
          for (size_t d = 0; d < dim; ++d)
            for (size_t i = 0; i < n_components; ++i) {
              if (blocks[i] != block) continue;
              for (size_t j = 0; j < n_components; ++j) {
                if (blocks[j] != block) continue;
                for (size_t c = 0; c < n_components; ++c)
                  restricted[d][i][j][c] = H[d][i][j][c];
              }
            }
          return restricted;
        }

        /**
         * @brief Run @p compute once per distinct block and hand each component its block's result.
         *
         * Components carrying `no_wave_speed` are skipped and keep the value-initialised entry.
         * `compute(block)` is called at most once per distinct id, so a model that declares one
         * block pays exactly what it paid before blocks existed.
         */
        template <size_t n_components, typename Result, typename ComputeFUN>
        std::array<Result, n_components> per_block(const std::array<int, n_components> &blocks,
                                                   const ComputeFUN &compute)
        {
          std::array<Result, n_components> per_component{};
          for (size_t i = 0; i < n_components; ++i) {
            if (blocks[i] < 0) continue;
            size_t source = i;
            for (size_t j = 0; j < i; ++j)
              if (blocks[j] == blocks[i]) {
                source = j;
                break;
              }
            per_component[i] = source == i ? compute(blocks[i]) : per_component[source];
          }
          return per_component;
        }

        /**
         * @brief The two traces' fluxes and the wave speeds of a face, see kt_flux_from_traces.
         */
        template <int dim, typename NumberType, size_t n_components> struct KTFluxData {
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> F_plus;
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> F_minus;
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> a_half;
        };

        /**
         * @brief The KT flux data of a face from the flux and its value jacobian dF/du on both traces: the fluxes
         * themselves, and one wave speed per block of wave_speed_blocks.
         */
        template <typename WaveSpeedStrategy, typename Model, typename NumberType, int dim, size_t n_components>
        KTFluxData<dim, NumberType, n_components>
        kt_flux_from_traces(const std::array<dealii::Tensor<1, dim, NumberType>, n_components> &F_plus,
                            const std::array<dealii::Tensor<1, dim, NumberType>, n_components> &F_minus,
                            const std::array<JacobianMatrix<NumberType, n_components>, dim> &J_plus,
                            const std::array<JacobianMatrix<NumberType, n_components>, dim> &J_minus,
                            const Model &model)
        {
          KTFluxData<dim, NumberType, n_components> result{};
          result.F_plus = F_plus;
          result.F_minus = F_minus;

          // One speed per block, from the flux jacobian restricted to that block, and a component
          // marked no_wave_speed gets none at all -- its numerical flux is then identically zero, so
          // the reconstruction, and its slope limiter, never reaches its row. See
          // AbstractModel::wave_speed_blocks for why a shared speed is wrong for a system that
          // mixes a conservation law with constraints.
          std::array<int, n_components> blocks;
          model.wave_speed_blocks(blocks);

          const auto a = per_block<n_components, std::array<NumberType, dim>>(blocks, [&](const int block) {
            return WaveSpeedStrategy::template compute_speeds<NumberType, dim, n_components>(
                restrict_jacobian_to_block<NumberType, dim, n_components>(J_plus, blocks, block),
                restrict_jacobian_to_block<NumberType, dim, n_components>(J_minus, blocks, block));
          });

          for (size_t d = 0; d < dim; ++d)
            for (size_t component = 0; component < n_components; ++component)
              result.a_half[component][d] = a[component][d];

          // A model that excludes a component and then writes a flux for it keeps the physical
          // average (F^+ + F^-)/2 but loses the dissipation that stabilises it -- a central flux on
          // a hyperbolic equation, which oscillates rather than failing outright. Catch the
          // contradiction here instead of leaving it in the solution.
          for (size_t i = 0; i < n_components; ++i)
            if (blocks[i] < 0)
              Assert(result.F_plus[i].norm() == NumberType(0) && result.F_minus[i].norm() == NumberType(0),
                     dealii::ExcMessage("A component marked no_wave_speed by wave_speed_blocks() wrote a flux."));

          return result;
        }

        template <int dim, typename NumberType, size_t n_components>
        std::array<dealii::Tensor<1, dim, NumberType>, n_components>
        compute_numerical_flux(const std::array<dealii::Tensor<1, dim, NumberType>, n_components> &F_plus,
                               const std::array<dealii::Tensor<1, dim, NumberType>, n_components> &F_minus,
                               const std::array<dealii::Tensor<1, dim, NumberType>, n_components> &a_half,
                               const std::array<NumberType, n_components> &u_plus,
                               const std::array<NumberType, n_components> &u_minus)
        {
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> H{};
          for (size_t c = 0; c < n_components; ++c) {
            H[c] = (F_plus[c] + F_minus[c]) * 0.5 - a_half[c] * (u_plus[c] - u_minus[c]) * 0.5;
          }
          return H;
        }

      } // namespace internal

      namespace internal
      {
        template <int dim, typename NumberType, size_t n_components> struct KTNumFluxJacobianData {
          std::array<SimpleMatrix<dealii::Tensor<1, dim, NumberType>, n_components>, 2> u{};
          std::array<SimpleMatrix<dealii::Tensor<1, dim, dealii::Tensor<1, dim, NumberType>>, n_components>, 2> grad{};
        };

        /**
         * @brief d(numerical flux)/d(trace values and trace gradients) of a face from the flux derivatives on both
         * traces (FluxDerivativeData: J, H, grad_J, mixed_H; F is not read).
         */
        template <typename WaveSpeedStrategy, typename Model, typename NumberType, int dim, size_t n_components>
        KTNumFluxJacobianData<dim, NumberType, n_components>
        kt_numflux_jacobian_from_derivatives(const FluxDerivativeData<NumberType, dim, n_components> &plus,
                                             const FluxDerivativeData<NumberType, dim, n_components> &minus,
                                             const std::array<NumberType, n_components> &u_plus,
                                             const std::array<NumberType, n_components> &u_minus, const Model &model)
        {
          // One wave speed per block, and its derivative, from the flux jacobian restricted to
          // that block. This has to mirror kt_flux_from_traces exactly -- a residual and a
          // jacobian that disagree about the dissipation are an inconsistent linearisation, which is
          // worse than either choice on its own. See AbstractModel::wave_speed_blocks.
          std::array<int, n_components> blocks;
          model.wave_speed_blocks(blocks);

          const auto a = per_block<n_components, std::array<NumberType, dim>>(blocks, [&](const int block) {
            return WaveSpeedStrategy::template compute_speeds<NumberType, dim, n_components>(
                restrict_jacobian_to_block<NumberType, dim, n_components>(plus.J, blocks, block),
                restrict_jacobian_to_block<NumberType, dim, n_components>(minus.J, blocks, block));
          });

          // Differentiate the selected physical wave speed with the AD flux Hessian.
          using SpeedDerivatives = std::pair<std::array<std::array<NumberType, n_components>, dim>,
                                             std::array<std::array<NumberType, n_components>, dim>>;
          const auto da = per_block<n_components, SpeedDerivatives>(blocks, [&](const int block) {
            return WaveSpeedStrategy::template compute_selected_speed_derivatives<NumberType, dim, n_components>(
                restrict_jacobian_to_block<NumberType, dim, n_components>(plus.J, blocks, block),
                restrict_jacobian_to_block<NumberType, dim, n_components>(minus.J, blocks, block),
                restrict_hessian_to_block<NumberType, dim, n_components>(plus.H, blocks, block),
                restrict_hessian_to_block<NumberType, dim, n_components>(minus.H, blocks, block));
          });

          // Assemble j_numflux
          KTNumFluxJacobianData<dim, NumberType, n_components> j_numflux{};

          for (size_t d = 0; d < dim; ++d) {
            for (size_t i = 0; i < n_components; ++i) {
              const NumberType du_i = u_plus[i] - u_minus[i];
              const auto &[da_plus, da_minus] = da[i];
              for (size_t c = 0; c < n_components; ++c) {
                const NumberType delta_ic = (i == c) ? NumberType(1) : NumberType(0);

                // dH_i^d / du_minus_c
                j_numflux.u[0](i, c)[d] = NumberType(0.5) * minus.J[d][i][c] + NumberType(0.5) * a[i][d] * delta_ic -
                                          NumberType(0.5) * du_i * da_minus[d][c];

                // dH_i^d / du_plus_c
                j_numflux.u[1](i, c)[d] = NumberType(0.5) * plus.J[d][i][c] - NumberType(0.5) * a[i][d] * delta_ic -
                                          NumberType(0.5) * du_i * da_plus[d][c];
              }
            }
          }

          for (size_t d_in = 0; d_in < dim; ++d_in) {
            const auto da_grad = per_block<n_components, SpeedDerivatives>(blocks, [&](const int block) {
              return WaveSpeedStrategy::template compute_selected_speed_derivatives<NumberType, dim, n_components>(
                  restrict_jacobian_to_block<NumberType, dim, n_components>(plus.J, blocks, block),
                  restrict_jacobian_to_block<NumberType, dim, n_components>(minus.J, blocks, block),
                  restrict_hessian_to_block<NumberType, dim, n_components>(plus.mixed_H[d_in], blocks, block),
                  restrict_hessian_to_block<NumberType, dim, n_components>(minus.mixed_H[d_in], blocks, block));
            });
            for (size_t d_out = 0; d_out < dim; ++d_out)
              for (size_t i = 0; i < n_components; ++i) {
                const NumberType du_i = u_plus[i] - u_minus[i];
                const auto &[da_grad_plus, da_grad_minus] = da_grad[i];
                for (size_t c = 0; c < n_components; ++c) {
                  j_numflux.grad[0](i, c)[d_out][d_in] = NumberType(0.5) * minus.grad_J[i][c][d_out][d_in] -
                                                         NumberType(0.5) * du_i * da_grad_minus[d_out][c];
                  j_numflux.grad[1](i, c)[d_out][d_in] = NumberType(0.5) * plus.grad_J[i][c][d_out][d_in] -
                                                         NumberType(0.5) * du_i * da_grad_plus[d_out][c];
                }
              }
          }

          return j_numflux;
        }

        /**
         * @brief Whether the model reads "fe_hessians" in model.source().
         *
         * Opt in with `static constexpr bool source_uses_hessians = true;` in the model. Off by default, so a
         * model that does not need curvature in its source pays nothing for it.
         */
        template <typename Model> consteval bool source_uses_hessians()
        {
          if constexpr (requires { Model::source_uses_hessians; })
            return Model::source_uses_hessians;
          else
            return false;
        }

        /**
         * @brief Diagonal second derivatives of the cell averages from the cell and its 2*dim face neighbours.
         *
         * Second derivative of the quadratic through the same three cell averages the gradient uses: with s
         * measured from the cell centre and u(s) = u_C + a s + b s^2 through (dx_1, u_1), (0, u_C), (dx_2, u_2),
         * u'' = 2b = 2 (du_2 - du_1) / (dx_2 - dx_1), where du_i are the one-sided slopes. Constant over the cell.
         *
         * Deliberately UNLIMITED: applying the limiter to a curvature would bias it toward zero exactly where the
         * solution is most curved. The price is that this is the noisiest quantity available near a steep front.
         *
         * Only the diagonal d^2/dx_d^2 entries are filled: the 2*dim stencil has no corner neighbours, so mixed
         * derivatives are not available. Across a physical boundary the neighbour slot holds the model's ghost.
         */
        template <int dim, typename NT, size_t n_components>
        std::array<dealii::Tensor<2, dim, NT>, n_components>
        stencil_hessians(const CellStencilData<dim, NT, n_components> &stencil)
        {
          std::array<dealii::Tensor<2, dim, NT>, n_components> hessians{};
          for (uint c = 0; c < n_components; ++c)
            for (int d = 0; d < dim; ++d) {
              const double dx_1 = stencil.neighbors.x[2 * d][d] - stencil.cell.x[d];
              const double dx_2 = stencil.neighbors.x[2 * d + 1][d] - stencil.cell.x[d];
              const NT du_1 = (stencil.neighbors.u[2 * d][c] - stencil.cell.u[c]) / dx_1;
              const NT du_2 = (stencil.neighbors.u[2 * d + 1][c] - stencil.cell.u[c]) / dx_2;
              hessians[c][d][d] = NT(2.) * (du_2 - du_1) / (dx_2 - dx_1);
            }
          return hessians;
        }

      } // namespace internal

      // Model_ keeps its second place and merely gains a default, rather than moving to the end.
      // Moving it would let an application that names a Reconstructor drop it, but it would also
      // force every *model-generic* discretization -- one Discretization built from a bare
      // ComponentDescriptor and shared across several models, which is how most of the regression
      // tests are written -- to spell out all four preceding arguments to reach the model.
      template <typename Discretization_,
                typename Model_ = typename DiFfRG::internal::assembler_model_of<Discretization_>::type,
                def::HasReconstructor Reconstructor_ =
                    def::TVDReconstructor<Discretization_::dim, def::MinModLimiter, double>,
                def::HasWaveSpeed WaveSpeedStrategy_ = MaxEigenvalueWaveSpeed,
                def::HasReconstructor JacobianReconstructor_ = Reconstructor_>
        requires MeshIsRectangular<typename Discretization_::Mesh>
      class Assembler
          : public DiFfRG::internal::AssemblerCore<
                Assembler<Discretization_, Model_, Reconstructor_, WaveSpeedStrategy_, JacobianReconstructor_>,
                Discretization_, Model_>
      {
        using Base = DiFfRG::internal::AssemblerCore<
            Assembler<Discretization_, Model_, Reconstructor_, WaveSpeedStrategy_, JacobianReconstructor_>,
            Discretization_, Model_>;

      public:
        using Discretization = Discretization_;
        using Model = Model_;
        static constexpr bool source_uses_hessians = internal::source_uses_hessians<Model>();
        using Reconstructor = Reconstructor_;
        using WaveSpeedStrategy = WaveSpeedStrategy_;
        using JacobianReconstructor = JacobianReconstructor_;
        using NumberType = typename Discretization::NumberType;
        using VectorType = typename Discretization::VectorType;
        using SparseMatrixType = typename Discretization::SparseMatrixType;

        using Components = typename Discretization::Components;
        static constexpr uint dim = Discretization::dim;
        static_assert(Reconstructor::dim == dim, "Reconstructor dimension must match the discretization dimension.");
        static_assert(JacobianReconstructor::dim == dim,
                      "JacobianReconstructor dimension must match the discretization dimension.");
        static constexpr uint n_components = Components::count_fe_functions(0);
        static constexpr uint n_faces = GeometryInfo<dim>::faces_per_cell;
        using GradientType = internal::GradientType<dim, NumberType, n_components>;
        using ThirdDerivativeType = internal::ThirdDerivativeType<dim, NumberType, n_components>;
        using Iterator = typename DoFHandler<Discretization::dim>::active_cell_iterator;
        using Point = dealii::Point<dim>;
        using Scratch = internal::ScratchData<dim, NumberType, n_components>;
        struct FaceJacobianDependencyCacheEntry {
          std::vector<types::global_dof_index> to_dofs;
          std::vector<types::global_dof_index> from_dofs;
        };
        struct FaceReconstructionDescriptor {
          unsigned int cell_index = 0;
          unsigned int face_index = 0;
          bool boundary = false;
          std::optional<unsigned int> neighbor_index;
          std::optional<unsigned int> neighbor_face_index;
          Point face_center;
        };
        struct CellTopologyCacheEntry {
          internal::CellStencilTopologyData<dim, n_components> stencil;
          /// Index into boundary_stencil_topologies of each physical boundary face, -1 for the others: the boundary
          /// topology is large (kilobytes in 2D) and only boundary faces have one.
          std::array<int, n_faces> boundary_stencil_slot{};
          std::array<FaceJacobianDependencyCacheEntry, n_faces> face_jacobian_dependencies{};
          FaceJacobianDependencyCacheEntry source_jacobian_dependencies{};
          std::vector<Point> quadrature_points;
          std::vector<NumberType> jxw;
          /// Local grid spacing, handed to the model through the flux and source ties.
          /// @see DiFfRG::internal::cell_width, DiFfRG::FV::KurganovTadmor::internal::flux_tie.
          double cell_width = 0.;
        };
        Assembler(Discretization &discretization, Model &model, const ConfigTree &config)
            : Base(discretization, model, config, "FV"), report_port(discretization.report_port()),
              triangulation(discretization.get_triangulation()),
              quadrature(1 + config.get_uint("/discretization/overintegration", 0)),
              diagnose_flux_conditioning(config.get_bool("/discretization/diagnose_flux_conditioning", false))
        {
          AssertThrow(fe.dofs_per_cell == n_components,
                      ExcMessage("FV Kurganov-Tadmor assembler expects one dof per component."));
          for (uint i = 0; i < n_components; ++i)
            local_component_of_dof[i] = fe.system_to_component_index(i).first;
          // About 16 MB of AD inputs per stacked evaluation: the derivative passes stream the traces through the AD
          // batch once per direction, and a batch that stays in cache is ~1.5x faster than one that does not (2D,
          // 160^2 cells: 59 vs 89 ms per jacobian evaluation). A model whose evaluate_batch launches GPU work may
          // want more points per call.
          constexpr size_t per_point =
              sizeof(autodiff::Real<2, NumberType>) * n_components * (2 + 2 * dim + dim * dim * dim);
          max_stacked_points = Base::read_max_stacked_points(config, per_point, size_t(16) << 20);

          reinit();
        }

        /// The solution handed to Model::abs_tolerances: the readout reconstruction at each cell centre.
        using AbsTolSolution = named_tuple<std::tuple<std::array<NumberType, n_components> &, GradientType &,
                                                      std::array<Tensor<2, dim, NumberType>, n_components> &>,
                                           StringSet<"fe_functions", "fe_derivatives", "fe_hessians">>;

        /**
         * @brief Per-dof absolute tolerances from Model::abs_tolerances, evaluated at every cell centre on the
         * readout reconstruction (values, reconstructed gradient, unlimited 3-point curvature).
         * @see AbstractAssembler::local_abs_tolerances. Serial vectors only: the reconstruction reads neighbour
         * cells, which a distributed vector would have to provide as ghosts.
         */
        virtual bool local_abs_tolerances(VectorType &atol, const VectorType &solution, double abs_tol,
                                          double rel_tol) const override
        {
          if constexpr (!def::HasAbsTolerances<Model, dim, AbsTolSolution, n_components> ||
                        !std::is_same_v<VectorType, dealii::Vector<NumberType>>)
            return false;
          else {
            this->reinit_vector(atol);
            std::array<double, n_components> cell_atol{};
            std::vector<types::global_dof_index> dofs(n_components);
            for (const auto &cell : dof_handler.active_cell_iterators()) {
              if (!cell->is_locally_owned()) continue;
              const Point x = cell->center();
              auto sol = reconstruct_readout_solution(cell, solution, x);
              cell_atol.fill(abs_tol);
              model.abs_tolerances(cell_atol, x, AbsTolSolution(std::tie(sol.values, sol.gradients, sol.hessians)),
                                   abs_tol, rel_tol);
              cell->get_dof_indices(dofs);
              for (uint i = 0; i < n_components; ++i)
                atol[dofs[i]] = cell_atol[local_component_of_dof[i]];
            }
            return true;
          }
        }

        virtual void reinit() override
        {
          Timer timer;
          this->reinit_common();
          this->build_mass_matrix(quadrature);

          // Hoisted out of probe_diffusion_flux_conditioning(): GridTools::diameter is COLLECTIVE
          // on a partitioned triangulation, and that function is entered conditionally
          // (diagnose_flux_conditioning, and only on the first residual assembly). A collective
          // behind a condition is a hang waiting for the condition to stop agreeing across ranks.
          // Here every rank arrives unconditionally, and the value cannot go stale because KT
          // refuses to run under mesh adaptivity.
          domain_diameter = GridTools::diameter(triangulation);

          rebuild_cell_topology_cache();
          rebuild_face_reconstruction_descriptors();
          rebuild_trace_maps();
          residual_reconstruction_cache.topology_initialized = false;
          jacobian_reconstruction_cache.topology_initialized = false;
          build_cached_jacobian_sparsity(sparsity_pattern_jacobian);

          timings_reinit.push_back(timer.wall_time());
        }

        virtual void jacobian_variables([[maybe_unused]] FullMatrix<NumberType> &jacobian,
                                        [[maybe_unused]] const VectorType &variables, const VectorType &) override {
          // Not assembled: KT treats the variables as frozen within a Newton step.
        };

        struct ReadoutSolution {
          std::array<NumberType, n_components> values{};
          GradientType gradients{};
          std::array<Tensor<2, dim, NumberType>, n_components> hessians{};
        };

        /**
         * @brief The solution at @p x in @p cell, for the extractors, readouts and abs tolerances: the cell value
         * reconstructed to @p x with the Reconstructor's gradient, and an unlimited 3-point curvature. Meant only for
         * single points like these: the curvature is not part of the scheme and must not be wired into the flux path.
         */
        ReadoutSolution reconstruct_readout_solution(const Iterator &cell, const VectorType &solution_global,
                                                     const Point &x) const
        {
          CellStencilData stencil;
          fill_cell_stencil(cell, solution_global, stencil);

          const auto gradients = Reconstructor::template compute_gradient<n_components>(
              stencil.cell.x, stencil.cell.u, stencil.neighbors.x, stencil.neighbors.u);

          ReadoutSolution solution;
          solution.values = internal::reconstruct_u(stencil.cell.u, stencil.cell.x, x, gradients);
          solution.gradients = Reconstructor::template compute_gradient_at_point<n_components>(
              stencil.cell.x, x, stencil.cell.u, stencil.neighbors.x, stencil.neighbors.u);

          solution.hessians = internal::stencil_hessians(stencil);
          return solution;
        }

        // The point evaluation of the extractors and readouts; see DiFfRG::internal::AssemblerCore.
        template <typename PotentialEvaluation> struct PointEvaluation : ReadoutSolution {
          PotentialEvaluation potential;
        };

        template <typename RawPotential>
        auto evaluate_at(const Point &x, const typename Base::CellIterator &cell, const VectorType &solution,
                         const RawPotential &raw_potential) const
        {
          PointEvaluation<decltype(evaluate_raw_potential(raw_potential, mapping, x))> evaluation;
          static_cast<ReadoutSolution &>(evaluation) = reconstruct_readout_solution(Iterator(cell), solution, x);
          evaluation.potential = evaluate_raw_potential(raw_potential, mapping, x);
          return evaluation;
        }

        template <typename Evaluation, typename Extractors_>
        static auto solution_tie(const Evaluation &e, const Extractors_ &extractors, const VectorType &variables)
        {
          return Base::e_tie(e.values, e.gradients, e.hessians, extractors, variables, e.potential.value,
                             e.potential.gradient, e.potential.mass_hessian);
        }

        /**
         * @brief The solution sample a model's extractor_point sees.
         *
         * Values are read straight off the dofs -- with one dof per cell they are the cell averages already -- and
         * gradients are recovered by central differences afterwards. Deliberately not the Reconstructor's limited
         * slopes: reconstructing per cell means a stencil fill over the whole mesh on every residual evaluation,
         * which is serial work that dominates the assembly (it cost a factor of four here). The FE path of
         * make_solution_sample() is no use either, since DG0 shape functions have zero gradient.
         */
        auto extractor_sample(const VectorType &solution_global) const
        {
          std::vector<types::global_dof_index> cell_dofs(n_components);
          auto sample = make_solution_sample<dim, NumberType>(dof_handler, mapping, n_components,
                                                              [&](const Iterator &cell, const Point &,
                                                                  std::vector<NumberType> &values,
                                                                  std::vector<Tensor<1, dim, NumberType>> &) {
                                                                cell->get_dof_indices(cell_dofs);
                                                                for (uint c = 0; c < n_components; ++c)
                                                                  values[c] = solution_global[cell_dofs[c]];
                                                              });
          sample.compute_central_difference_gradients();
          return sample;
        }

        virtual void mass(VectorType &mass, const VectorType &solution_global, const VectorType &solution_global_dot,
                          NumberType weight) override
        {
          scatter_rows(mass, [&](const size_t k, Scratch &scratch, CellRows &local) {
            const auto &cell = cells[k];
            const auto &geometry = cell_topology_cache[cell->active_cell_index()];
            local.reinit(cell, false);
            fill_constant_quadrature_values(cell, solution_global, solution_global_dot, scratch);
            std::array<NumberType, n_components> mass_values{};
            for (size_t q = 0; q < geometry.quadrature_points.size(); ++q) {
              model.mass(mass_values, geometry.quadrature_points[q], scratch.solution_values[q],
                         scratch.solution_dot_values[q]);
              for (uint i = 0; i < n_components; ++i)
                local.residual(i) += weight * geometry.jxw[q] * mass_values[local_component_of_dof[i]];
            }
          });
        }

        using CellData = internal::CellData<dim, NumberType, n_components>;
        using NeighborData = internal::NeighborData<dim, NumberType, n_components>;
        using CellStencilData = internal::CellStencilData<dim, NumberType, n_components>;
        using CellGeometryDofs = internal::CellGeometryDofs<dim, n_components>;
        using CellStencilTopologyData = internal::CellStencilTopologyData<dim, n_components>;
        using FaceReconstructionState = internal::FaceReconstructionState<dim, NumberType, n_components>;
        using SolutionReconstructionCache = internal::SolutionReconstructionCache<dim, NumberType, n_components>;
        template <typename NT> using CellStencilDataT = internal::CellStencilData<dim, NT, n_components>;

        struct AssemblyFaceGeometryProvider {
          dealii::Tensor<1, dim> normal(const Iterator &cell, const unsigned int face_index) const
          {
            return Assembler::face_normal_from_cell(cell, face_index);
          }

          double jxw(const Iterator &cell, const unsigned int face_index) const
          {
            return Assembler::face_jxw(cell, face_index);
          }
        };

        auto make_assembly_context_view(const SolutionReconstructionCache &cache) const
        {
          using FaceRange =
              FaceAssemblyViewRange<dim, NumberType, n_components, Iterator, AssemblyFaceGeometryProvider>;
          using CellRange = CellAssemblyViewRange<dim, NumberType, n_components, Iterator>;
          return AssemblyContextView<FaceRange, CellRange>(
              FaceRange(dof_handler.begin_active(), dof_handler.end(), cache, AssemblyFaceGeometryProvider{}),
              CellRange(dof_handler.begin_active(), dof_handler.end(), cache));
        }

        auto make_assembly_context_view(SolutionReconstructionCache &&cache) const = delete;

        template <HasAssemblyContextView Context>
        void run_fv_kt_pre_assembly_hook(const AssemblyStage stage, const Context &context)
        {
          dispatch_fv_kt_pre_assembly(model, stage, context);
        }

        static bool is_physical_boundary_face(const Iterator &cell, const unsigned int face_index)
        {
          return cell->at_boundary(face_index);
        }

        static Iterator face_neighbor(const Iterator &cell, const unsigned int face_index)
        {
          return cell->neighbor(face_index);
        }

        void fill_cell_data_from_topology(const CellGeometryDofs &topology, const VectorType &solution_global,
                                          CellData &data) const
        {
          internal::fill_cell_data_from_topology<dim, NumberType, n_components>(topology, solution_global, data);
        }

        template <int boundary_dim, typename BoundaryNumberType>
        internal::BoundaryStencilData<boundary_dim, BoundaryNumberType, n_components>
        build_boundary_stencil_from_cache_impl(const Iterator &cell, const unsigned int boundary_face_no,
                                               const VectorType &solution_global) const
        {
          const auto &topology = boundary_stencil_topology(cell->active_cell_index(), boundary_face_no).primary;
          return internal::fill_boundary_stencil_from_topology<BoundaryNumberType, boundary_dim, n_components>(
              topology, solution_global);
        }

        template <typename BoundaryNumberType>
        internal::BoundaryStencilData<dim, BoundaryNumberType, n_components>
        build_boundary_stencil_from_cache(const Iterator &cell, const unsigned int boundary_face_no,
                                          const VectorType &solution_global) const
        {
          return build_boundary_stencil_from_cache_impl<dim, BoundaryNumberType>(cell, boundary_face_no,
                                                                                 solution_global);
        }

        template <typename BoundaryNumberType>
        internal::BoundaryReconstructionStencilData<dim, BoundaryNumberType, n_components>
        build_boundary_reconstruction_stencil_from_cache(const Iterator &cell, const unsigned int boundary_face_no,
                                                         const VectorType &solution_global) const
        {
          const auto &topology = boundary_stencil_topology(cell->active_cell_index(), boundary_face_no);
          return internal::fill_boundary_reconstruction_stencil_from_topology<BoundaryNumberType, dim, n_components>(
              topology, solution_global);
        }

        void fill_cell_stencil(const Iterator &cell, const VectorType &solution_global, CellStencilData &stencil) const
        {
          initialize_cell_stencil_topology(cell->active_cell_index(), stencil);
          refresh_cell_stencil_values(cell->active_cell_index(), solution_global, stencil);
        }

        CellStencilDataT<autodiff::Real<1, NumberType>>
        tag_cell_stencil_dofs_from_cache(const Iterator &cell, const CellStencilData &cell_stencil,
                                         const VectorType &solution_global, const types::global_dof_index dof_j) const
        {
          auto tagged_stencil = internal::tag_cell_stencil_dofs<dim, NumberType, n_components>(cell_stencil, dof_j);

          for (const auto face_index : cell->face_indices()) {
            if (!is_physical_boundary_face(cell, face_index)) continue;

            const auto boundary_stencil =
                build_boundary_stencil_from_cache<NumberType>(cell, face_index, solution_global);
            auto tagged_boundary_stencil =
                internal::tag_boundary_stencil_dofs<dim, NumberType, n_components>(boundary_stencil, dof_j);
            internal::populate_boundary_neighbor_from_model_stencil(tagged_boundary_stencil, tagged_stencil, face_index,
                                                                    cell_stencil.face_centers[face_index], model);
          }

          return tagged_stencil;
        }

        unsigned int find_neighbor_face(const Iterator &cell, const Iterator &neighbor) const
        {
          for (const auto neighbor_face_index : neighbor->face_indices()) {
            if (is_physical_boundary_face(neighbor, neighbor_face_index)) continue;
            if (face_neighbor(neighbor, neighbor_face_index) == cell) return neighbor_face_index;
          }
          AssertThrow(false, ExcMessage("Could not find reciprocal KT neighbor face."));
          return 0;
        }

        template <def::HasReconstructor ActiveReconstructor>
        void rebuild_solution_reconstruction_cache(const VectorType &solution_global,
                                                   SolutionReconstructionCache &cache) const
        {
          ensure_solution_reconstruction_cache_shape(cache);
          for (auto &valid_faces : cache.face_reconstruction_valid)
            valid_faces.fill(false);

          refresh_solution_reconstruction_cache_values(solution_global, cache);

          // Each face is written by exactly one descriptor (both of its orientations), so they run in parallel.
          tbb::parallel_for(
              tbb::blocked_range<size_t>(0, face_reconstruction_descriptors.size()),
              [&](const tbb::blocked_range<size_t> &r) {
                for (size_t k = r.begin(); k != r.end(); ++k) {
                  const auto &descriptor = face_reconstruction_descriptors[k];
                  const auto cell_index = descriptor.cell_index;
                  const auto face_index = descriptor.face_index;
                  const auto &x_q = descriptor.face_center;

                  if (descriptor.boundary) {
                    const auto &topology = boundary_stencil_topology(cell_index, face_index);
                    auto boundary_stencil =
                        internal::fill_boundary_reconstruction_stencil_from_topology<NumberType, dim, n_components>(
                            topology, solution_global);
                    cache.face_reconstructions[cell_index][face_index] =
                        internal::compute_boundary_face_reconstruction_state<ActiveReconstructor>(
                            boundary_stencil, cache.cell_stencils[cell_index], x_q, model);
                    cache.face_reconstruction_valid[cell_index][face_index] = true;
                    continue;
                  }

                  Assert(descriptor.neighbor_index.has_value(), ExcInternalError());
                  Assert(descriptor.neighbor_face_index.has_value(), ExcInternalError());
                  const auto neighbor_index = *descriptor.neighbor_index;
                  const auto neighbor_face_index = *descriptor.neighbor_face_index;
                  AssertIndexRange(neighbor_index, cache.cell_stencils.size());
                  AssertIndexRange(neighbor_face_index, n_faces);

                  const auto state = internal::compute_interior_face_reconstruction_state<ActiveReconstructor>(
                      cache.cell_stencils[cell_index], cache.cell_stencils[neighbor_index], x_q);
                  cache.face_reconstructions[cell_index][face_index] = state;
                  cache.face_reconstruction_valid[cell_index][face_index] = true;
                  cache.face_reconstructions[neighbor_index][neighbor_face_index] =
                      internal::reverse_face_reconstruction(state);
                  cache.face_reconstruction_valid[neighbor_index][neighbor_face_index] = true;
                }
              });
        }

        void ensure_solution_reconstruction_cache_shape(SolutionReconstructionCache &cache) const
        {
          const auto n_active_cells = triangulation.n_active_cells();
          if (cache.cell_stencils.size() != n_active_cells) {
            cache.cell_stencils.resize(n_active_cells);
            cache.topology_initialized = false;
          }
          if (cache.face_reconstructions.size() != n_active_cells) cache.face_reconstructions.resize(n_active_cells);
          if (cache.face_reconstruction_valid.size() != n_active_cells)
            cache.face_reconstruction_valid.resize(n_active_cells);
          if (!cache.topology_initialized) initialize_solution_reconstruction_cache_topology(cache);
        }

        void initialize_solution_reconstruction_cache_topology(SolutionReconstructionCache &cache) const
        {
          for (unsigned int cell_index = 0; cell_index < cache.cell_stencils.size(); ++cell_index)
            initialize_cell_stencil_topology(cell_index, cache.cell_stencils[cell_index]);
          cache.topology_initialized = true;
        }

        void initialize_cell_stencil_topology(const unsigned int cell_index, CellStencilData &stencil) const
        {
          const auto &topology = cell_topology_cache[cell_index].stencil;
          stencil.boundary_ids = topology.boundary_ids;
          stencil.face_centers = topology.face_centers;
          stencil.cell.x = topology.cell.x;
          stencil.cell.dof_indices = topology.cell.dof_indices;
          stencil.neighbors.x = topology.neighbors.x;
          stencil.neighbors.dof_indices = topology.neighbors.dof_indices;
        }

        void refresh_solution_reconstruction_cache_values(const VectorType &solution_global,
                                                          SolutionReconstructionCache &cache) const
        {
          // The model's apply_boundary_stencil runs on several threads at once here.
          tbb::parallel_for(tbb::blocked_range<unsigned int>(0, cache.cell_stencils.size()),
                            [&](const tbb::blocked_range<unsigned int> &r) {
                              for (unsigned int cell_index = r.begin(); cell_index != r.end(); ++cell_index)
                                refresh_cell_stencil_values(cell_index, solution_global,
                                                            cache.cell_stencils[cell_index]);
                            });
        }

        void refresh_cell_stencil_values(const unsigned int cell_index, const VectorType &solution_global,
                                         CellStencilData &stencil) const
        {
          const auto &topology = cell_topology_cache[cell_index].stencil;
          for (unsigned int i = 0; i < n_components; ++i)
            stencil.cell.u[i] = solution_global(topology.cell.dof_indices[i]);

          for (unsigned int face_index = 0; face_index < n_faces; ++face_index) {
            if (topology.boundary_ids[face_index] == numbers::invalid_boundary_id) {
              for (unsigned int i = 0; i < n_components; ++i) {
                const auto dof = topology.neighbors.dof_indices[face_index][i];
                stencil.neighbors.u[face_index][i] = solution_global(dof);
              }
              continue;
            }

            auto boundary_stencil = internal::fill_boundary_stencil_from_topology<NumberType, dim, n_components>(
                boundary_stencil_topology(cell_index, face_index).primary, solution_global);
            internal::populate_boundary_neighbor_from_model_stencil(boundary_stencil, stencil, face_index,
                                                                    topology.face_centers[face_index], model);
          }
        }

        void rebuild_face_reconstruction_descriptors()
        {
          face_reconstruction_descriptors.clear();
          face_reconstruction_descriptors.reserve(triangulation.n_active_cells() * n_faces);

          // Every cell, not just the owned ones: an owned cell's faces may be described from its neighbour's
          // side, and the `neighbor_index < cell_index` tiebreak below needs both sides
          // present to pick a side consistently.
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            const auto cell_index = cell->active_cell_index();
            for (const auto face_index : cell->face_indices()) {
              FaceReconstructionDescriptor descriptor;
              descriptor.cell_index = cell_index;
              descriptor.face_index = face_index;
              descriptor.boundary = is_physical_boundary_face(cell, face_index);
              descriptor.face_center = cell_topology_cache[cell_index].stencil.face_centers[face_index];
              if (descriptor.boundary) {
                face_reconstruction_descriptors.push_back(descriptor);
                continue;
              }

              const auto neighbor = face_neighbor(cell, face_index);
              const auto neighbor_index = neighbor->active_cell_index();
              if (neighbor_index < cell_index) continue;
              descriptor.neighbor_index = neighbor_index;
              descriptor.neighbor_face_index = find_neighbor_face(cell, neighbor);
              face_reconstruction_descriptors.push_back(descriptor);
            }
          }
        }

        const FaceReconstructionState &get_cached_face_reconstruction(const SolutionReconstructionCache &cache,
                                                                      const Iterator &cell,
                                                                      const unsigned int face_index) const
        {
          const auto cell_index = cell->active_cell_index();
          AssertIndexRange(cell_index, cache.face_reconstructions.size());
          AssertIndexRange(face_index, n_faces);
          AssertThrow(cache.face_reconstruction_valid[cell_index][face_index],
                      ExcMessage("KT face reconstruction cache entry was not initialized."));
          return cache.face_reconstructions[cell_index][face_index];
        }

        /**
         * @brief One-shot conditioning check on the model's diffusion flux.
         *
         * The FV residual differences the diffusion flux between a cell's faces. Any part of
         * the flux that does *not* depend on the gradient therefore cancels analytically —
         * but not in floating point, where it leaves its round-off, |F|·eps, behind. When that
         * gradient-independent baseline dwarfs the gradient-dependent part, the residual loses
         * the corresponding number of digits, and an implicit solver asked for a tight
         * tolerance will grind its step size down trying to resolve pure noise.
         *
         * This is the standard failure mode for fRG loop integrals at large RG scale: a flux
         * c·k^n·g(k² + ∂u) carries an O(k^(n-1)) baseline, so the relative noise is ~eps·k²/Δ(∂u)
         * and blows up as the UV cutoff grows.
         *
         * The fix belongs in the model, not here: return the flux with its zero-gradient value
         * already subtracted, F - F|_{∂u = 0}, written in a cancellation-free form. Subtracting
         * a gradient-independent constant leaves the residual unchanged (a constant flux has
         * zero divergence and cancels on the ghost side of boundary faces too).
         */
        template <typename ExtractorArray>
        void probe_diffusion_flux_conditioning(const SolutionReconstructionCache &reconstruction_cache,
                                               const ExtractorArray &extractors, const VectorType &variables) const
        {
          // Probe the MODEL, not the current data. Using the solution's actual gradient fails
          // whenever the flow starts from flat initial data: the local variation is then ~0 and
          // every flux looks baseline-dominated. Instead perturb the gradient by a synthetic
          // scale built from the solution magnitude and the domain size, which is what the
          // gradient will grow to once the flow develops.
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> D_actual{};
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> D_baseline{};
          const ThirdDerivativeType zero_third{};
          const GradientType zero_grad{};

          const double domain_size = std::max(domain_diameter, 1e-300);
          std::array<double, n_components> max_baseline{};
          std::array<double, n_components> max_variation{};
          double u_scale = 0.0;
          unsigned int sampled = 0;

          // First pass: the scale of the solution itself.
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (sampled >= flux_conditioning_max_samples) break;
            for (unsigned int f = 0; f < n_faces; ++f) {
              if (cell->at_boundary(f)) continue;
              const auto &reconstruction = get_cached_face_reconstruction(reconstruction_cache, cell, f);
              for (size_t c = 0; c < n_components; ++c)
                u_scale = std::max(u_scale, std::abs(static_cast<double>(reconstruction.diffusion_u_minus[c])));
              ++sampled;
              break;
            }
          }

          // A unit fallback keeps the probe meaningful for a flow starting from u == 0.
          const NumberType probe_gradient = static_cast<NumberType>(std::max(u_scale, 1.0) / domain_size);
          GradientType probe_grad{};
          for (size_t c = 0; c < n_components; ++c)
            for (int d = 0; d < dim; ++d)
              probe_grad[c][d] = probe_gradient;

          sampled = 0;
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (sampled >= flux_conditioning_max_samples) break;
            for (unsigned int f = 0; f < n_faces; ++f) {
              if (cell->at_boundary(f)) continue;
              const auto x_q = cell->face(f)->center();
              const auto &reconstruction = get_cached_face_reconstruction(reconstruction_cache, cell, f);
              const double probe_width = get_cell_topology(cell).cell_width;
              model.diffusion_flux(D_actual, x_q,
                                   internal::diffusion_flux_tie(reconstruction.diffusion_u_minus, probe_grad,
                                                                zero_third, extractors, variables, probe_width));
              model.diffusion_flux(D_baseline, x_q,
                                   internal::diffusion_flux_tie(reconstruction.diffusion_u_minus, zero_grad, zero_third,
                                                                extractors, variables, probe_width));
              for (size_t c = 0; c < n_components; ++c) {
                max_baseline[c] = std::max(max_baseline[c], static_cast<double>(D_baseline[c].norm()));
                max_variation[c] =
                    std::max(max_variation[c], static_cast<double>((D_actual[c] - D_baseline[c]).norm()));
              }
              ++sampled;
              break; // one face per cell is plenty
            }
          }

          for (size_t c = 0; c < n_components; ++c) {
            if (!(max_variation[c] > 0.0) || !std::isfinite(max_baseline[c])) continue;

            const double relative_noise =
                max_baseline[c] / max_variation[c] * std::numeric_limits<NumberType>::epsilon();
            if (relative_noise <= flux_conditioning_warn_threshold) continue;

            const auto message = spdlog::fmt_lib::format(
                "FV/KT: the diffusion flux of component {} is dominated by a gradient-independent baseline: "
                "max |F(du=0)| = {:.3e} over the mesh, but max |F - F(du=0)| = only {:.3e}. Differencing it "
                "across faces leaves a relative round-off of {:.1e} in the residual, which caps the accuracy any "
                "implicit solver can reach (and will crush its step size if the tolerance is tighter). Subtract "
                "the zero-gradient value analytically inside diffusion_flux().",
                c, max_baseline[c], max_variation[c], relative_noise);

            report_port.warn("{}", message);
          }
        }

        void fill_constant_quadrature_values(const Iterator &cell, const VectorType &solution_global,
                                             const VectorType &solution_global_dot, Scratch &scratch_data) const
        {
          fill_cell_data_from_topology(get_cell_topology(cell).stencil.cell, solution_global,
                                       scratch_data.cell_stencil.cell);
          for (auto &values : scratch_data.solution_values)
            for (uint c = 0; c < n_components; ++c)
              values[c] = scratch_data.cell_stencil.cell.u[c];

          for (auto &values_dot : scratch_data.solution_dot_values)
            for (uint c = 0; c < n_components; ++c)
              values_dot[c] = solution_global_dot(scratch_data.cell_stencil.cell.dof_indices[c]);
        }

        /**
         * @brief The "fe_derivatives" slot of the source tuple, i.e. the gradient model.source() sees at x_q.
         *
         * The same reconstruction the readout path and the face fluxes use. At the default single
         * quadrature point -- the cell centre -- compute_gradient_at_point falls through to the limiter, so
         * this is the scheme's own limited slope; with overintegration it becomes one-sided towards x_q.
         *
         * Boundary cells need no special case: the stencil's neighbour slot across a physical boundary face
         * already holds the model's ghost value (see refresh_cell_stencil_values).
         */
        template <def::HasReconstructor ActiveReconstructor>
        static GradientType source_gradient(const CellStencilData &stencil, const Point &x_q)
        {
          return ActiveReconstructor::template compute_gradient_at_point<n_components>(
              stencil.cell.x, x_q, stencil.cell.u, stencil.neighbors.x, stencil.neighbors.u);
        }

        static Tensor<1, dim> face_normal_from_cell(const Iterator &cell, const unsigned int face_no)
        {
          const auto face_offset = cell->face(face_no)->center() - cell->center();
          const auto norm = face_offset.norm();
          AssertThrow(norm > 0., ExcMessage("Degenerate FV face normal."));
          return face_offset / norm;
        }

        static double face_jxw(const Iterator &cell, const unsigned int face_no)
        {
          if constexpr (dim == 1)
            return 1.;
          else
            return cell->face(face_no)->measure();
        }

        /// The extractor values the model sees, as the assembler passes them.
        using Extractors = std::array<NumberType, Components::count_extractors()>;
        using FluxTraces =
            internal::TraceBatch<dim, NumberType, n_components, Extractors, VectorType, internal::TraceKind::flux>;
        using DiffusionTraces =
            internal::TraceBatch<dim, NumberType, n_components, Extractors, VectorType, internal::TraceKind::diffusion>;
        using SourcePoints = internal::TraceBatch<dim, NumberType, n_components, Extractors, VectorType,
                                                  source_uses_hessians ? internal::TraceKind::source_hessians
                                                                       : internal::TraceKind::source>;
        using TraceFlux = std::array<Tensor<1, dim, NumberType>, n_components>;
        using TraceJacobian = std::array<internal::JacobianMatrix<NumberType, n_components>, dim>;
        using FluxDerivatives = internal::FluxDerivativeData<NumberType, dim, n_components>;
        using DiffusionSideJacobian = internal::DiffusionSideJacobian<dim, NumberType, n_components>;
        using SourceJacobian = PointJacobian<dim, n_components, n_components, Components::count_extractors()>;

        using PhaseTimes = AssemblyPhaseTimes;

        /**
         * @brief Number the faces the owned cells need: each face once, in face_reconstruction_descriptors order,
         * its trace on the descriptor's cell side (u^-) as point 2p and the other side (u^+) as point 2p + 1; and the
         * owned cells for the source.
         */
        void rebuild_trace_maps()
        {
          const auto n_active = triangulation.n_active_cells();
          cells.reinit(dof_handler, discretization.get_constraints());
          active_cells.clear();
          for (const auto &cell : dof_handler.active_cell_iterators())
            active_cells.push_back(cell);
          std::vector<bool> owned(n_active, false);
          for (const auto &cell : cells.all())
            owned[cell->active_cell_index()] = true;

          trace_faces.clear();
          std::array<int, n_faces> none;
          none.fill(-1);
          trace_point_of.assign(n_active, none);
          for (unsigned int k = 0; k < face_reconstruction_descriptors.size(); ++k) {
            const auto &face = face_reconstruction_descriptors[k];
            if (!owned[face.cell_index] && (face.boundary || !owned[*face.neighbor_index])) continue;
            const int p = trace_faces.size();
            trace_faces.push_back(k);
            trace_point_of[face.cell_index][face.face_index] = 2 * p;
            if (!face.boundary) trace_point_of[*face.neighbor_index][*face.neighbor_face_index] = 2 * p + 1;
          }
        }

        /// The model's pre-assembly hook, if it has one; the context view is only built then.
        void run_pre_assembly(const AssemblyStage stage, const SolutionReconstructionCache &cache)
        {
          if constexpr (HasFVKTAssemblyHook<Model, decltype(make_assembly_context_view(cache))>)
            run_fv_kt_pre_assembly_hook(stage, make_assembly_context_view(cache));
        }

        /// Phase 1: the advection and diffusion traces of every face the owned cells need.
        void gather_traces(const SolutionReconstructionCache &cache, const Extractors &extractors,
                           const VectorType &variables)
        {
          const size_t n_traces = 2 * trace_faces.size();
          flux_traces.reinit(n_traces);
          diffusion_traces.reinit(n_traces);
          tbb::parallel_for(
              tbb::blocked_range<size_t>(0, trace_faces.size()), [&](const tbb::blocked_range<size_t> &r) {
                for (size_t p = r.begin(); p != r.end(); ++p) {
                  const auto &face = face_reconstruction_descriptors[trace_faces[p]];
                  const auto &state = cache.face_reconstructions[face.cell_index][face.face_index];
                  const double width_minus = cell_topology_cache[face.cell_index].cell_width;
                  const double width_plus =
                      face.boundary ? width_minus : cell_topology_cache[*face.neighbor_index].cell_width;
                  const auto store = [&](auto &batch, const size_t i, const auto &u, const auto &grad,
                                         const double width) {
                    for (uint c = 0; c < n_components; ++c) {
                      batch.value(c, i) = u[c];
                      for (uint d = 0; d < dim; ++d)
                        batch.derivative(c, d, i) = grad[c][d];
                    }
                    for (uint d = 0; d < dim; ++d)
                      batch.coordinate(d, i) = face.face_center[d];
                    batch.width(i) = width;
                  };
                  const auto store_third = [&](const size_t i, const auto &third) {
                    for (uint c = 0; c < n_components; ++c)
                      for (uint d0 = 0; d0 < dim; ++d0)
                        for (uint d1 = 0; d1 < dim; ++d1)
                          for (uint d2 = 0; d2 < dim; ++d2)
                            diffusion_traces.third_derivative(c, d0, d1, d2, i) = third[c][d0][d1][d2];
                  };
                  store(flux_traces, 2 * p, state.u_minus, state.face_grad_minus, width_minus);
                  store(flux_traces, 2 * p + 1, state.u_plus, state.face_grad_plus, width_plus);
                  store(diffusion_traces, 2 * p, state.diffusion_u_minus, state.diffusion_grad_minus, width_minus);
                  store(diffusion_traces, 2 * p + 1, state.diffusion_u_plus, state.diffusion_grad_plus, width_plus);
                  if constexpr (dim == 1) {
                    store_third(2 * p, state.third_derivatives_minus);
                    store_third(2 * p + 1, state.third_derivatives_plus);
                  }
                }
              });
          flux_traces.set_shared(extractors, variables);
          diffusion_traces.set_shared(extractors, variables);
        }

        /// Phase 1: the solution the source sees at every quadrature point of the owned cells.
        template <def::HasReconstructor ActiveReconstructor>
        void gather_sources(const SolutionReconstructionCache &cache, const Extractors &extractors,
                            const VectorType &variables)
        {
          const size_t n_q = quadrature.size();
          source_points.reinit(cells.size() * n_q);
          tbb::parallel_for(tbb::blocked_range<size_t>(0, cells.size()), [&](const tbb::blocked_range<size_t> &r) {
            for (size_t k = r.begin(); k != r.end(); ++k) {
              const auto cell_index = cells[k]->active_cell_index();
              const auto &stencil = cache.cell_stencils[cell_index];
              const auto &geometry = cell_topology_cache[cell_index];
              for (size_t q = 0; q < n_q; ++q) {
                const size_t i = k * n_q + q;
                const auto &x_q = geometry.quadrature_points[q];
                const auto gradients = source_gradient<ActiveReconstructor>(stencil, x_q);
                for (uint c = 0; c < n_components; ++c) {
                  source_points.value(c, i) = stencil.cell.u[c];
                  for (uint d = 0; d < dim; ++d)
                    source_points.derivative(c, d, i) = gradients[c][d];
                }
                if constexpr (source_uses_hessians) {
                  const auto hessians = internal::stencil_hessians(stencil);
                  for (uint c = 0; c < n_components; ++c)
                    for (uint d1 = 0; d1 < dim; ++d1)
                      for (uint d2 = 0; d2 < dim; ++d2)
                        source_points.hessian(c, d1, d2, i) = hessians[c][d1][d2];
                }
                for (uint d = 0; d < dim; ++d)
                  source_points.coordinate(d, i) = x_q[d];
                source_points.width(i) = geometry.cell_width;
              }
            }
          });
          source_points.set_shared(extractors, variables);
        }

        /**
         * @brief Phase 2 of the residual: the advection flux and its value jacobian dF/du at every trace (the
         * jacobian sets the wave speed), the diffusion flux at every trace and the source at every cell point, each
         * as batched model evaluations.
         */
        void evaluate_residual_terms()
        {
          const auto evaluate = [&](auto &out, const auto &batch) { model.evaluate_batch(out, batch); };
          const size_t n_traces = flux_traces.size();
          internal::flux_value_jacobians<n_components>(value_seed_workspace, flux_traces, 0, n_traces,
                                                       max_stacked_points, evaluate, trace_F, trace_J);

          BatchOutput<dim, NumberType, n_components> diffusion;
          diffusion.reinit(n_traces, Term::diffusion_flux);
          model.evaluate_batch(diffusion, diffusion_traces);
          trace_D.resize(n_traces);
          tbb::parallel_for(tbb::blocked_range<size_t>(0, n_traces), [&](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); ++i)
              for (uint c = 0; c < n_components; ++c)
                for (uint d = 0; d < dim; ++d)
                  trace_D[i][c][d] = diffusion.diffusion_flux(c, d)[i];
          });

          source_values.reinit(source_points.size(), Term::source);
          model.evaluate_batch(source_values, source_points);
        }

        /// The flux and the diffusion flux directions evaluate_trace_jacobians() differentiates along per trace.
        static constexpr std::array<size_t, 2> n_flux_diffusion_directions{
            internal::n_flux_derivative_directions<dim, n_components>,
            internal::n_diffusion_derivative_directions<dim, n_components>};

        /**
         * @brief Phase 2 of the jacobian for the traces [begin, end): the flux derivatives the numerical flux
         * jacobian needs (internal::flux_derivatives) and the diffusion flux jacobian
         * (internal::diffusion_flux_jacobians), into trace_derivatives and trace_diffusion_jacobians at index i -
         * begin.
         */
        void evaluate_trace_jacobians(const size_t begin, const size_t end)
        {
          const auto evaluate = [&](auto &out, const auto &batch) { model.evaluate_batch(out, batch); };
          internal::flux_derivatives<n_components>(flux_derivative_workspace, flux_traces, begin, end,
                                                   max_stacked_points, evaluate, trace_derivatives);
          internal::diffusion_flux_jacobians<n_components>(diffusion_derivative_workspace, diffusion_traces, begin, end,
                                                           max_stacked_points, evaluate, trace_diffusion_jacobians);
        }

        /// Phase 2 of the jacobian at the cell points: the source jacobian.
        void evaluate_source_jacobians()
        {
          constexpr uint n = n_components;
          // The source. Extractors and variables stay frozen within a Newton step.
          DiFfRG::internal::parallel_assign(source_jacobians, source_points.size(), SourceJacobian{});
          if constexpr (DiFfRG::internal::has_batch_ad_jacobians<Model>)
            DiFfRG::internal::seed_stacked_jacobian<n, 1>(
                [&](auto &out, const auto &ad, size_t) { model.evaluate_batch(out, ad[0]); },
                [&](const size_t i, uint) -> auto & { return source_jacobians[i]; },
                std::array<const SourcePoints *, 1>{&source_points}, max_stacked_points, Term::source, source_workspace,
                /*extractor_seeds = */ false);
          else
            DiFfRG::internal::for_each_point(source_points, [&](const size_t i, const auto &x, const auto &sol) {
              auto &J = source_jacobians[i];
              model.template jacobian_source<0, 0>(J.j_source, x, sol);
              model.template jacobian_source_grad<1>(J.j_grad_source, x, sol);
              if constexpr (source_uses_hessians) model.template jacobian_source_hess<2>(J.j_hess_source, x, sol);
            });
        }

        /// The KT numerical flux H of trace face p, as the face's u^- cell sees it.
        TraceFlux numerical_flux(const size_t p, const FaceReconstructionState &state) const
        {
          const size_t minus = 2 * p, plus = 2 * p + 1;
          const auto [F_plus, F_minus, a_half] =
              internal::kt_flux_from_traces<WaveSpeedStrategy, Model, NumberType, dim, n_components>(
                  trace_F[plus], trace_F[minus], trace_J[plus], trace_J[minus], model);
          return internal::compute_numerical_flux(F_plus, F_minus, a_half, state.u_plus, state.u_minus);
        }

        /// The diffusion flux D of trace face p: the average of its two traces' diffusion fluxes.
        TraceFlux diffusion_flux(const size_t p) const
        {
          TraceFlux D;
          for (uint c = 0; c < n_components; ++c)
            D[c] = NumberType(0.5) * (trace_D[2 * p][c] + trace_D[2 * p + 1][c]);
          return D;
        }

        /// One owned cell's contribution to its own rows: the residual, or the jacobian as a square cell block plus
        /// rectangular blocks whose columns are a reconstruction stencil's dofs.
        struct CellRows {
          struct Block {
            const std::vector<types::global_dof_index> *columns = nullptr;
            FullMatrix<NumberType> values;
          };
          std::vector<types::global_dof_index> dofs;
          Vector<NumberType> residual;
          FullMatrix<NumberType> cell_block;
          std::vector<Block> blocks;
          uint n_blocks = 0;

          void reinit(const Iterator &cell, const bool with_matrix)
          {
            dofs.resize(n_components);
            cell->get_dof_indices(dofs);
            residual.reinit(n_components);
            if (with_matrix) cell_block.reinit(n_components, n_components);
            n_blocks = 0;
          }
          /// A zeroed block with the given columns.
          FullMatrix<NumberType> &next_block(const std::vector<types::global_dof_index> &columns)
          {
            if (n_blocks == blocks.size()) blocks.emplace_back();
            auto &block = blocks[n_blocks++];
            block.columns = &columns;
            block.values.reinit(n_components, columns.size());
            return block.values;
          }
        };
        struct RowScratch {
          RowScratch(const dealii::Quadrature<dim> &quadrature) : scratch(quadrature) {}
          Scratch scratch;
          CellRows rows;
        };

        void insert_rows(VectorType &global, const CellRows &local) const
        {
          discretization.get_constraints().distribute_local_to_global(local.residual, local.dofs, global);
        }
        void insert_rows(SparseMatrixType &global, const CellRows &local) const
        {
          const auto &constraints = discretization.get_constraints();
          constraints.distribute_local_to_global(local.cell_block, local.dofs, global);
          for (uint b = 0; b < local.n_blocks; ++b)
            constraints.distribute_local_to_global(local.blocks[b].values, local.dofs, *local.blocks[b].columns,
                                                   global);
        }

        /// The batches point to the extractors and variables of the residual() or jacobian() call that gathered
        /// them; this drops those pointers when the call ends.
        struct SharedScope {
          Assembler &assembler;
          ~SharedScope()
          {
            assembler.flux_traces.clear_shared();
            assembler.diffusion_traces.clear_shared();
            assembler.source_points.clear_shared();
            assembler.value_seed_workspace.ad.clear_shared();
            assembler.flux_derivative_workspace.stacked.ad.clear_shared();
            assembler.diffusion_derivative_workspace.ad.clear_shared();
          }
        };

        /// Phase 3: assemble(k, scratch, rows) for every owned cell k, each into its own rows only.
        template <typename Global, typename Assemble> void scatter_rows(Global &global, const Assemble &assemble)
        {
          cells.scatter(
              global, row_scratch, &RowScratch::rows, row_buffer,
              [&](const size_t k, RowScratch &s, CellRows &local) { assemble(k, s.scratch, local); },
              [&](const CellRows &local) { insert_rows(global, local); });
        }

        /// The trace face p of face f of a cell, and +1 if the cell is its u^- side, -1 if it is its u^+ side.
        std::pair<size_t, NumberType> face_of(const unsigned int cell_index, const unsigned int f) const
        {
          const int point = trace_point_of[cell_index][f];
          Assert(point >= 0, ExcInternalError());
          return {size_t(point / 2), point % 2 == 0 ? NumberType(1) : NumberType(-1)};
        }

        /**
         * @brief Phase 3 of the residual, per face: weight * JxW * (H + D) . n as the face's u^- cell sees it. Its
         * u^+ cell gets the negative.
         *
         * Sign convention: the advection numerical flux H (from flux()) and the diffusion flux D (from
         * diffusion_flux()) are SUMMED, as CG and LLFFlux sum all contributions into one flux. A diffusion flux must
         * therefore decrease with the gradient for forward diffusion, e.g. -nu * du for the heat equation.
         */
        void face_residuals(const SolutionReconstructionCache &cache, const NumberType weight)
        {
          face_values.resize(trace_faces.size());
          tbb::parallel_for(
              tbb::blocked_range<size_t>(0, trace_faces.size()), [&](const tbb::blocked_range<size_t> &r) {
                for (size_t p = r.begin(); p != r.end(); ++p) {
                  const auto &d = face_reconstruction_descriptors[trace_faces[p]];
                  const auto &cell = active_cells[d.cell_index];
                  const auto n = face_normal_from_cell(cell, d.face_index);
                  const auto JxW = face_jxw(cell, d.face_index);
                  const auto H = numerical_flux(p, cache.face_reconstructions[d.cell_index][d.face_index]);
                  const auto D = diffusion_flux(p);
                  for (uint c = 0; c < n_components; ++c)
                    face_values[p][c] = weight * JxW * (scalar_product(H[c], n) + scalar_product(D[c], n));
                }
              });
        }

        /**
         * @brief Phases 2 and 3 of the jacobian at the faces, chunk by chunk: the model's flux and diffusion flux
         * derivatives at the chunk's traces (evaluate_trace_jacobians), then per face d(weight * JxW * (H + D) .
         * n)/du_j in the rows of its u^- cell, for the dofs u_j its reconstruction stencils read
         * (face_jacobian_dependencies of that side); its u^+ cell gets the negative. A chunk holds as many faces as one
         * stacked evaluation, so the per-trace derivatives never exist for more than one chunk.
         */
        void face_jacobians(const SolutionReconstructionCache &cache, const VectorType &solution_global,
                            const NumberType weight, DiFfRG::internal::PhaseTimer &phase)
        {
          const size_t n_faces = trace_faces.size();
          const size_t faces_per_chunk = std::max<size_t>(
              1, max_stacked_points / (2 * std::max(n_flux_diffusion_directions[0], n_flux_diffusion_directions[1])));
          face_blocks.resize(n_faces);
          // map() is collective, the faces are rank-local; apply_boundary_stencil may be a model callback, too.
          const NoMapsHere no_maps_during_assembly;
          for (size_t f0 = 0; f0 < n_faces; f0 += faces_per_chunk) {
            const size_t f1 = std::min(n_faces, f0 + faces_per_chunk);
            evaluate_trace_jacobians(2 * f0, 2 * f1);
            phase.lap(&PhaseTimes::evaluate);
            tbb::parallel_for(tbb::blocked_range<size_t>(f0, f1), [&](const tbb::blocked_range<size_t> &r) {
              auto &scratch = row_scratch.local().scratch;
              for (size_t p = r.begin(); p != r.end(); ++p)
                face_jacobian(p, 2 * f0, cache, solution_global, weight, scratch, face_blocks[p]);
            });
            phase.lap(&PhaseTimes::scatter);
          }
        }

        /// The chain rule of face p; the derivatives of its traces sit at 2 p - first_trace in the trace stores.
        void face_jacobian(const size_t p, const size_t first_trace, const SolutionReconstructionCache &cache,
                           const VectorType &solution_global, const NumberType weight, Scratch &scratch_data,
                           FullMatrix<NumberType> &block) const
        {
          const auto &d = face_reconstruction_descriptors[trace_faces[p]];
          const auto &cell = active_cells[d.cell_index];
          const unsigned int f = d.face_index;
          const auto x_q = cell->face(f)->center();
          const auto JxW = face_jxw(cell, f);
          const auto n_face = face_normal_from_cell(cell, f);
          const auto &from_dofs = cell_topology_cache[d.cell_index].face_jacobian_dependencies[f].from_dofs;
          const uint n_from = from_dofs.size();

          auto &reconstructed_deriv = scratch_data.reconstructed_derivatives;
          auto &diffusion_deriv = scratch_data.diffusion_derivatives;
          for (auto &derivatives : reconstructed_deriv)
            derivatives.resize(n_from);
          for (auto &derivatives : diffusion_deriv)
            derivatives.resize(n_from);

          const auto &cell_stencil = cache.cell_stencils[d.cell_index];
          if (!d.boundary) {
            const auto &ncell = active_cells[*d.neighbor_index];
            const auto &ncell_stencil = cache.cell_stencils[*d.neighbor_index];
            const auto &cell_data = cell_stencil.cell;
            const auto &ncell_data = ncell_stencil.cell;
            for (uint j = 0; j < n_from; ++j) {
              const auto cell_stencil_tagged =
                  tag_cell_stencil_dofs_from_cache(cell, cell_stencil, solution_global, from_dofs[j]);
              const auto ncell_stencil_tagged =
                  tag_cell_stencil_dofs_from_cache(ncell, ncell_stencil, solution_global, from_dofs[j]);

              // d(u^-)/d(u_j) from the cell side, d(u^+)/d(u_j) from the neighbour side.
              reconstructed_deriv[0][j].u =
                  internal::reconstruct_u_derivative<JacobianReconstructor, dim, NumberType, n_components>(
                      cell_stencil_tagged.cell.u, cell_data.x, x_q, cell_stencil.neighbors.x,
                      cell_stencil_tagged.neighbors.u);
              reconstructed_deriv[0][j].grad =
                  JacobianReconstructor::template compute_gradient_at_point_derivative<n_components>(
                      cell_data.x, x_q, cell_stencil_tagged.cell.u, cell_stencil.neighbors.x,
                      cell_stencil_tagged.neighbors.u);
              reconstructed_deriv[1][j].u =
                  internal::reconstruct_u_derivative<JacobianReconstructor, dim, NumberType, n_components>(
                      ncell_stencil_tagged.cell.u, ncell_data.x, x_q, ncell_stencil.neighbors.x,
                      ncell_stencil_tagged.neighbors.u);
              reconstructed_deriv[1][j].grad =
                  JacobianReconstructor::template compute_gradient_at_point_derivative<n_components>(
                      ncell_data.x, x_q, ncell_stencil_tagged.cell.u, ncell_stencil.neighbors.x,
                      ncell_stencil_tagged.neighbors.u);

              if constexpr (dim == 1) {
                const auto third_derivative_stencil_ad =
                    internal::make_interior_third_derivative_stencil(cell_stencil_tagged, ncell_stencil_tagged, x_q);
                const auto third_derivatives =
                    JacobianReconstructor::template compute_third_derivatives_at_face_derivative<n_components>(
                        third_derivative_stencil_ad.x, third_derivative_stencil_ad.u);
                reconstructed_deriv[0][j].third_derivatives = third_derivatives;
                reconstructed_deriv[1][j].third_derivatives = third_derivatives;
              }

              const auto diffusion_face_derivatives =
                  internal::extract_diffusion_face_derivatives<dim, NumberType, n_components>(
                      internal::compute_diffusion_face_state(cell_stencil_tagged, ncell_stencil_tagged));
              diffusion_deriv[0][j] = diffusion_face_derivatives[0];
              diffusion_deriv[1][j] = diffusion_face_derivatives[1];
            }
          } else {
            const auto boundary_stencil =
                build_boundary_reconstruction_stencil_from_cache<NumberType>(cell, f, solution_global);
            for (uint j = 0; j < n_from; ++j) {
              const auto boundary_stencil_ad =
                  internal::tag_boundary_reconstruction_stencil_dofs<dim, NumberType, n_components>(boundary_stencil,
                                                                                                    from_dofs[j]);
              const auto cell_stencil_ad =
                  tag_cell_stencil_dofs_from_cache(cell, cell_stencil, solution_global, from_dofs[j]);
              const auto [physical_stencil_ad, ghost_stencil_ad] =
                  internal::make_model_boundary_reconstruction_side_stencils(boundary_stencil_ad, cell_stencil_ad, x_q,
                                                                             model);
              reconstructed_deriv[0][j].u =
                  internal::reconstruct_u_derivative<JacobianReconstructor, dim, NumberType, n_components>(
                      physical_stencil_ad.cell.u, physical_stencil_ad.cell.x, x_q, physical_stencil_ad.neighbors.x,
                      physical_stencil_ad.neighbors.u);
              reconstructed_deriv[1][j].u =
                  internal::reconstruct_u_derivative<JacobianReconstructor, dim, NumberType, n_components>(
                      ghost_stencil_ad.cell.u, ghost_stencil_ad.cell.x, x_q, ghost_stencil_ad.neighbors.x,
                      ghost_stencil_ad.neighbors.u);
              reconstructed_deriv[0][j].grad =
                  JacobianReconstructor::template compute_gradient_at_point_derivative<n_components>(
                      physical_stencil_ad.cell.x, x_q, physical_stencil_ad.cell.u, physical_stencil_ad.neighbors.x,
                      physical_stencil_ad.neighbors.u);
              reconstructed_deriv[1][j].grad =
                  JacobianReconstructor::template compute_gradient_at_point_derivative<n_components>(
                      ghost_stencil_ad.cell.x, x_q, ghost_stencil_ad.cell.u, ghost_stencil_ad.neighbors.x,
                      ghost_stencil_ad.neighbors.u);

              if constexpr (dim == 1) {
                auto third_derivative_boundary_stencil_ad = boundary_stencil_ad.primary;
                internal::apply_boundary_reconstruction_stencil(third_derivative_boundary_stencil_ad, cell_stencil_ad,
                                                                model);
                const auto third_derivative_stencil_ad =
                    internal::make_boundary_third_derivative_stencil(third_derivative_boundary_stencil_ad);
                const auto third_derivatives =
                    JacobianReconstructor::template compute_third_derivatives_at_face_derivative<n_components>(
                        third_derivative_stencil_ad.x, third_derivative_stencil_ad.u);
                reconstructed_deriv[0][j].third_derivatives = third_derivatives;
                reconstructed_deriv[1][j].third_derivatives = third_derivatives;
              }

              // The boundary diffusion flux is differentiated as one pair of corrected face-gradient operators,
              // with the same physical/ghost side labels as the advective boundary reconstruction.
              const auto diffusion_face_derivatives =
                  internal::extract_diffusion_face_derivatives<dim, NumberType, n_components>(
                      internal::compute_diffusion_face_state(physical_stencil_ad, ghost_stencil_ad));
              diffusion_deriv[0][j] = diffusion_face_derivatives[0];
              diffusion_deriv[1][j] = diffusion_face_derivatives[1];
            }
          }

          const auto &state = cache.face_reconstructions[d.cell_index][f];
          const auto j_numflux =
              internal::kt_numflux_jacobian_from_derivatives<WaveSpeedStrategy, Model, NumberType, dim, n_components>(
                  trace_derivatives[2 * p + 1 - first_trace], trace_derivatives[2 * p - first_trace], state.u_plus,
                  state.u_minus, model);
          const std::array<const DiffusionSideJacobian *, 2> j_diffusion{
              &trace_diffusion_jacobians[2 * p - first_trace], &trace_diffusion_jacobians[2 * p + 1 - first_trace]};

          // Chain rule: (dH/d trace) * (d trace/du_j) + (dD/d trace) * (d trace/du_j), both sides, dotted with n.
          block.reinit(n_components, n_from);
          for (uint i = 0; i < n_components; ++i)
            for (uint j = 0; j < n_from; ++j) {
              NumberType contribution{};
              for (size_t side = 0; side < 2; ++side)
                for (size_t c = 0; c < n_components; ++c) {
                  contribution += scalar_product(j_numflux.u[side](i, c), n_face) * reconstructed_deriv[side][j].u[c];
                  contribution += scalar_product(j_diffusion[side]->u(i, c), n_face) * diffusion_deriv[side][j].u[c];
                  for (size_t d_in = 0; d_in < dim; ++d_in)
                    for (size_t d_out = 0; d_out < dim; ++d_out)
                      contribution +=
                          n_face[d_out] *
                          (j_numflux.grad[side](i, c)[d_out][d_in] * reconstructed_deriv[side][j].grad[c][d_in] +
                           j_diffusion[side]->grad(i, c)[d_out][d_in] * diffusion_deriv[side][j].grad[c][d_in]);
                  if constexpr (dim == 1) // the only dimension with reconstructed third derivatives
                    contribution += j_diffusion[side]->third_derivatives(i, c)[0][0][0][0] * n_face[0] *
                                    reconstructed_deriv[side][j].third_derivatives[c][0][0][0];
                }
              block(i, j) = weight * JxW * contribution;
            }
        }

        /// Phase 3 of the residual for owned cell k: mass and source at its quadrature points, and its faces.
        void assemble_cell_residual(const size_t k, Scratch &scratch, CellRows &local,
                                    const VectorType &solution_global, const VectorType &solution_global_dot,
                                    const NumberType weight, const NumberType weight_mass) const
        {
          const auto &cell = cells[k];
          const auto cell_index = cell->active_cell_index();
          const auto &geometry = cell_topology_cache[cell_index];
          local.reinit(cell, false);
          fill_constant_quadrature_values(cell, solution_global, solution_global_dot, scratch);

          const size_t first_point = k * quadrature.size();
          std::array<NumberType, n_components> mass{};
          for (size_t q = 0; q < geometry.quadrature_points.size(); ++q) {
            model.mass(mass, geometry.quadrature_points[q], scratch.solution_values[q], scratch.solution_dot_values[q]);
            for (uint i = 0; i < n_components; ++i) {
              const auto c = local_component_of_dof[i];
              local.residual(i) +=
                  geometry.jxw[q] * (weight_mass * mass[c] + weight * source_values.source(c)[first_point + q]);
            }
          }
          for (const auto f : cell->face_indices()) {
            const auto [p, sign] = face_of(cell_index, f);
            for (uint i = 0; i < n_components; ++i)
              local.residual(i) += sign * face_values[p][local_component_of_dof[i]];
          }
        }

        /// Phase 3 of the jacobian for owned cell k: source and mass at its quadrature points, and its faces.
        void assemble_cell_jacobian(const size_t k, Scratch &scratch, CellRows &local,
                                    const SolutionReconstructionCache &cache, const VectorType &solution_global,
                                    const VectorType &solution_global_dot, const NumberType weight,
                                    const NumberType alpha, const NumberType beta) const
        {
          const auto &cell = cells[k];
          const auto cell_index = cell->active_cell_index();
          const auto &geometry = cell_topology_cache[cell_index];
          local.reinit(cell, true);
          fill_constant_quadrature_values(cell, solution_global, solution_global_dot, scratch);
          const auto &stencil = cache.cell_stencils[cell_index];

          // The source's gradient dependence reaches the whole 2*dim stencil through the reconstruction, so it gets
          // a rectangular block, like the faces.
          const auto &source_from = geometry.source_jacobian_dependencies.from_dofs;
          auto &source_block = local.next_block(source_from);
          auto &source_grad_deriv = scratch.source_gradient_derivatives;
          source_grad_deriv.resize(source_from.size());
          auto &source_hess_deriv = scratch.source_hessian_derivatives;
          if constexpr (source_uses_hessians) source_hess_deriv.resize(source_from.size());

          SimpleMatrix<NumberType, n_components> j_mass;
          SimpleMatrix<NumberType, n_components> j_mass_dot;
          const size_t first_point = k * quadrature.size();
          for (size_t q = 0; q < geometry.quadrature_points.size(); ++q) {
            const auto &x_q = geometry.quadrature_points[q];
            model.template jacobian_mass<0>(j_mass, x_q, scratch.solution_values[q], scratch.solution_dot_values[q]);
            model.template jacobian_mass<1>(j_mass_dot, x_q, scratch.solution_values[q],
                                            scratch.solution_dot_values[q]);
            // No extractor blocks: KT never assembles extractor_cell_jacobian and jacobian_variables is a no-op, so
            // extractors and variables are frozen within a Newton step.
            const auto &Js = source_jacobians[first_point + q];
            for (uint i = 0; i < n_components; ++i) {
              const auto ci = local_component_of_dof[i];
              for (uint j = 0; j < n_components; ++j) {
                const auto cj = local_component_of_dof[j];
                local.cell_block(i, j) += geometry.jxw[q] * (weight * Js.j_source(ci, cj) + alpha * j_mass_dot(ci, cj) +
                                                             beta * j_mass(ci, cj));
              }
            }

            // d(source)/d(u_j) through the gradient (and hessian): seed one stencil dof at a time.
            for (uint j = 0; j < source_from.size(); ++j) {
              const auto stencil_tagged =
                  tag_cell_stencil_dofs_from_cache(cell, stencil, solution_global, source_from[j]);
              source_grad_deriv[j] = JacobianReconstructor::template compute_gradient_at_point_derivative<n_components>(
                  stencil.cell.x, x_q, stencil_tagged.cell.u, stencil.neighbors.x, stencil_tagged.neighbors.u);
              if constexpr (source_uses_hessians) {
                // The hessian is linear in the stencil values, so forward AD through it is exact.
                const auto hessians_tagged = internal::stencil_hessians(stencil_tagged);
                for (uint c = 0; c < n_components; ++c)
                  for (uint d = 0; d < dim; ++d)
                    source_hess_deriv[j][c][d][d] = autodiff::derivative(hessians_tagged[c][d][d]);
              }
            }
            for (uint i = 0; i < n_components; ++i) {
              const auto ci = local_component_of_dof[i];
              for (uint j = 0; j < source_from.size(); ++j) {
                NumberType contribution{};
                for (uint c = 0; c < n_components; ++c)
                  for (uint d = 0; d < dim; ++d) {
                    contribution += Js.j_grad_source(ci, c)[d] * source_grad_deriv[j][c][d];
                    if constexpr (source_uses_hessians)
                      contribution += Js.j_hess_source(ci, c)[d][d] * source_hess_deriv[j][c][d][d];
                  }
                source_block(i, j) += weight * geometry.jxw[q] * contribution;
              }
            }
          }

          for (const auto f : cell->face_indices()) {
            const auto [p, sign] = face_of(cell_index, f);
            const auto &d = face_reconstruction_descriptors[trace_faces[p]];
            auto &block =
                local.next_block(cell_topology_cache[d.cell_index].face_jacobian_dependencies[d.face_index].from_dofs);
            for (uint i = 0; i < n_components; ++i)
              for (uint j = 0; j < block.n(); ++j)
                block(i, j) = sign * face_blocks[p](local_component_of_dof[i], j);
          }
        }

        virtual void residual(VectorType &residual, const VectorType &solution_global, NumberType weight,
                              const VectorType &solution_global_dot, NumberType weight_mass,
                              const VectorType &variables = VectorType()) override
        {
          // Find the EoM and extract whatever data is needed for the model, as the FEM assemblers do. Beyond
          // filling `extracted_data` this is what gives a model the chance to refresh whatever internal state its
          // flux depends on (interpolators, self-consistently solved anomalous dimensions, ...) before the fluxes
          // are evaluated. Models without extractors are unaffected.
          DiFfRG::internal::PhaseTimer phase(residual_times);
          Extractors extracted{};
          if constexpr (Components::count_extractors() > 0)
            this->extract(extracted, solution_global, variables, true, false, true);
          const SharedScope shared_scope{*this};
          phase.lap(&PhaseTimes::extract);

          rebuild_solution_reconstruction_cache<Reconstructor>(solution_global, residual_reconstruction_cache);
          const auto &reconstruction_cache = residual_reconstruction_cache;
          run_pre_assembly(AssemblyStage::residual, reconstruction_cache);
          gather_traces(reconstruction_cache, extracted, variables);
          gather_sources<Reconstructor>(reconstruction_cache, extracted, variables);
          phase.lap(&PhaseTimes::gather);

          // Runs on the first residual assembly only, so it costs nothing on the hot path. The first assembly is
          // also the most informative sample: for an fRG flow it happens at k = Lambda, where a flux baseline is
          // largest.
          if (diagnose_flux_conditioning && !flux_conditioning_probed) {
            flux_conditioning_probed = true;
            const NoMapsHere no_maps_during_assembly; // the probe calls the model's per-point diffusion flux
            probe_diffusion_flux_conditioning(reconstruction_cache, extracted, variables);
          }

          {
            const NoMapsHere no_maps_during_assembly; // map() is collective, the points are rank-local
            evaluate_residual_terms();
          }
          phase.lap(&PhaseTimes::evaluate);

          face_residuals(reconstruction_cache, weight);
          scatter_rows(residual, [&](const size_t k, Scratch &scratch, CellRows &local) {
            assemble_cell_residual(k, scratch, local, solution_global, solution_global_dot, weight, weight_mass);
          });
          phase.lap(&PhaseTimes::scatter);
          timings_residual.push_back(phase.finish());
        }

        virtual void jacobian_mass(SparseMatrixType &jacobian, const VectorType &solution_global,
                                   const VectorType &solution_global_dot, NumberType alpha = 1.,
                                   NumberType beta = 1.) override
        {
          Timer timer;
          scatter_rows(jacobian, [&](const size_t k, Scratch &scratch, CellRows &local) {
            const auto &cell = cells[k];
            const auto &geometry = cell_topology_cache[cell->active_cell_index()];
            local.reinit(cell, true);
            fill_constant_quadrature_values(cell, solution_global, solution_global_dot, scratch);
            SimpleMatrix<NumberType, n_components> j_mass;
            SimpleMatrix<NumberType, n_components> j_mass_dot;
            for (size_t q = 0; q < geometry.quadrature_points.size(); ++q) {
              const auto &x_q = geometry.quadrature_points[q];
              model.template jacobian_mass<0>(j_mass, x_q, scratch.solution_values[q], scratch.solution_dot_values[q]);
              model.template jacobian_mass<1>(j_mass_dot, x_q, scratch.solution_values[q],
                                              scratch.solution_dot_values[q]);
              for (uint i = 0; i < n_components; ++i)
                for (uint j = 0; j < n_components; ++j) {
                  const auto ci = local_component_of_dof[i], cj = local_component_of_dof[j];
                  local.cell_block(i, j) += geometry.jxw[q] * (alpha * j_mass_dot(ci, cj) + beta * j_mass(ci, cj));
                }
            }
          });
          timings_jacobian.push_back(timer.wall_time());
        }

        virtual void jacobian(SparseMatrixType &jacobian, const VectorType &solution_global, NumberType weight,
                              const VectorType &solution_global_dot, NumberType alpha, NumberType beta,
                              const VectorType &variables = VectorType()) override
        {
          // See residual(): keep the model's extractor-driven state consistent with the point the jacobian is
          // linearised about. The extractor jacobian contribution itself is not assembled here, so extractors are
          // treated as frozen w.r.t. the FE solution within a Newton step -- fine for the IDA/explicit split where
          // Variables are stepped explicitly, but it is why jacobian_variables is still a no-op.
          DiFfRG::internal::PhaseTimer phase(jacobian_times);
          Extractors extracted{};
          if constexpr (Components::count_extractors() > 0)
            this->extract(extracted, solution_global, variables, true, false, true);
          const SharedScope shared_scope{*this};
          phase.lap(&PhaseTimes::extract);

          rebuild_solution_reconstruction_cache<JacobianReconstructor>(solution_global, jacobian_reconstruction_cache);
          const auto &reconstruction_cache = jacobian_reconstruction_cache;
          run_pre_assembly(AssemblyStage::jacobian, reconstruction_cache);
          gather_traces(reconstruction_cache, extracted, variables);
          gather_sources<JacobianReconstructor>(reconstruction_cache, extracted, variables);
          phase.lap(&PhaseTimes::gather);

          {
            const NoMapsHere no_maps_during_assembly; // map() is collective, the points are rank-local
            evaluate_source_jacobians();
          }
          phase.lap(&PhaseTimes::evaluate);

          face_jacobians(reconstruction_cache, solution_global, weight, phase);

          scatter_rows(jacobian, [&](const size_t k, Scratch &scratch, CellRows &local) {
            assemble_cell_jacobian(k, scratch, local, reconstruction_cache, solution_global, solution_global_dot,
                                   weight, alpha, beta);
          });
          phase.lap(&PhaseTimes::scatter);
          timings_jacobian.push_back(phase.finish());
        }

        void refinement_indicator([[maybe_unused]] Vector<double> &indicator,
                                  [[maybe_unused]] const VectorType &solution_global)
        {
        }

        template <typename DoFContainer>
        static void append_dofs(std::vector<types::global_dof_index> &target, const DoFContainer &source)
        {
          target.insert(target.end(), source.begin(), source.end());
        }

        template <typename DoFContainer>
        static void append_valid_dofs(std::vector<types::global_dof_index> &target, const DoFContainer &source)
        {
          for (const auto dof : source)
            if (dof != numbers::invalid_dof_index) target.push_back(dof);
        }

        static void sort_unique_dofs(std::vector<types::global_dof_index> &dofs)
        {
          std::sort(dofs.begin(), dofs.end());
          dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
        }

        /**
         * @brief The dofs @p cell's reconstruction stencil reads: the cell and its 2*dim face neighbours, or, across a
         * physical boundary face, the dofs the model's ghost value there is built from.
         */
        void append_cell_stencil_dofs(std::vector<types::global_dof_index> &from_dofs, const Iterator &cell) const
        {
          const auto &cell_topology = cell_topology_cache[cell->active_cell_index()].stencil;
          append_dofs(from_dofs, cell_topology.cell.dof_indices);
          for (const auto face_index : cell->face_indices()) {
            if (is_physical_boundary_face(cell, face_index))
              append_boundary_reconstruction_dofs(from_dofs, cell->active_cell_index(), face_index);
            else
              append_valid_dofs(from_dofs, cell_topology.neighbors.dof_indices[face_index]);
          }
        }

        void append_boundary_reconstruction_dofs(std::vector<types::global_dof_index> &from_dofs,
                                                 const unsigned int cell_index, const unsigned int face_index) const
        {
          const auto append_boundary_stencil_dofs =
              [&](const internal::BoundaryStencilTopologyData<dim, n_components> &topology) {
                for (const auto &dofs : topology.dof_indices)
                  append_valid_dofs(from_dofs, dofs);
              };

          const auto &topology = boundary_stencil_topology(cell_index, face_index);
          append_boundary_stencil_dofs(topology.primary);
          for (size_t face = 0; face < topology.tangential_ghost_neighbor_valid.size(); ++face) {
            if (topology.tangential_ghost_neighbor_valid[face])
              append_boundary_stencil_dofs(topology.tangential_ghost_neighbors[face]);
            if (topology.corner_tangential_stencil_valid[face]) {
              for (const auto &corner_stencil : topology.corner_tangential_stencils[face])
                append_boundary_stencil_dofs(corner_stencil);
            }
          }
        }

        void build_face_jacobian_dependency_cache(const Iterator &cell, const unsigned int face_index,
                                                  FaceJacobianDependencyCacheEntry &dependencies) const
        {
          const auto &cell_topology = cell_topology_cache[cell->active_cell_index()].stencil;

          dependencies.to_dofs.clear();
          append_dofs(dependencies.to_dofs, cell_topology.cell.dof_indices);
          if (!is_physical_boundary_face(cell, face_index))
            append_dofs(dependencies.to_dofs, cell_topology.neighbors.dof_indices[face_index]);

          // The two traces of a face are reconstructed from the stencils of the two cells (a boundary face's ghost
          // side from the boundary stencil), so a row reaches jacobian_stencil_radius = 2 cells. A first-order
          // reconstruction reads the two cells only.
          dependencies.from_dofs = dependencies.to_dofs;
          if (is_physical_boundary_face(cell, face_index))
            append_boundary_reconstruction_dofs(dependencies.from_dofs, cell->active_cell_index(), face_index);
          if constexpr (JacobianReconstructor::jacobian_stencil_radius > 1) {
            append_cell_stencil_dofs(dependencies.from_dofs, cell);
            if (!is_physical_boundary_face(cell, face_index))
              append_cell_stencil_dofs(dependencies.from_dofs, face_neighbor(cell, face_index));
          }
          sort_unique_dofs(dependencies.from_dofs);
        }

        /**
         * @brief The dofs the nonlocal part of the source jacobian writes to and reads from.
         *
         * Radius 1, tighter than the face dependencies: the gradient handed to model.source() is
         * reconstructed from the cell and its 2*dim face neighbours only. Across a physical boundary face
         * that neighbour is a model ghost, so the boundary stencil's dofs take its place.
         */
        void build_source_jacobian_dependency_cache(const Iterator &cell,
                                                    FaceJacobianDependencyCacheEntry &dependencies) const
        {
          const auto &cell_topology = cell_topology_cache[cell->active_cell_index()].stencil;

          dependencies.to_dofs.clear();
          append_dofs(dependencies.to_dofs, cell_topology.cell.dof_indices);

          dependencies.from_dofs.clear();
          append_cell_stencil_dofs(dependencies.from_dofs, cell);
          sort_unique_dofs(dependencies.from_dofs);
        }

        void build_cached_jacobian_sparsity(get_type::SparsityPattern<SparseMatrixType> &sparsity_pattern) const
        {
          DynamicSparsityPattern dsp(discretization.get_locally_relevant_dofs());
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            const auto &topology = get_cell_topology(cell);
            const auto &cell_dofs = topology.stencil.cell.dof_indices;
            for (const auto row : cell_dofs)
              for (const auto column : cell_dofs)
                dsp.add(row, column);

            const auto add_block = [&](const FaceJacobianDependencyCacheEntry &dependencies) {
              for (const auto row : dependencies.to_dofs)
                for (const auto column : dependencies.from_dofs)
                  if (row != numbers::invalid_dof_index && column != numbers::invalid_dof_index) dsp.add(row, column);
            };
            for (const auto face_index : cell->face_indices())
              add_block(topology.face_jacobian_dependencies[face_index]);
            // Subsumed by the face blocks in practice, but the source jacobian must not depend on that.
            add_block(topology.source_jacobian_dependencies);
          }
          finalize_la_sparsity<SparseMatrixType>(dsp, sparsity_pattern, discretization.get_locally_owned_dofs(),
                                                 discretization.get_locally_relevant_dofs(),
                                                 discretization.get_communicator());
        }

        void fill_boundary_topology(internal::BoundaryStencilTopologyData<dim, n_components> &boundary_topology,
                                    const Iterator &cell, const unsigned int boundary_face_no,
                                    std::vector<types::global_dof_index> &dof_indices) const
        {
          using namespace def::BoundaryStencilIndex;
          boundary_topology.lower_boundary = boundary_face_no % 2 == 0;
          boundary_topology.cell_face = boundary_face_no;
          boundary_topology.face_center = cell->face(boundary_face_no)->center();
          boundary_topology.ghost_center = boundary_topology.lower_boundary ? lower_inner : upper_inner;
          boundary_topology.ghost_left = boundary_topology.lower_boundary ? lower_outer : physical_cell;
          boundary_topology.ghost_right = boundary_topology.lower_boundary ? physical_cell : upper_outer;
          for (auto &dofs : boundary_topology.dof_indices)
            dofs.fill(numbers::invalid_dof_index);

          cell->get_dof_indices(dof_indices);
          boundary_topology.x[physical_cell] = cell->center();
          for (uint i = 0; i < n_components; ++i)
            boundary_topology.dof_indices[physical_cell][i] = dof_indices[i];

          const auto interior_face = GeometryInfo<dim>::opposite_face[boundary_face_no];
          AssertThrow(!is_physical_boundary_face(cell, interior_face),
                      ExcMessage("KT boundary stencil requires at least two interior cells behind the boundary face."));

          auto neighbor = face_neighbor(cell, interior_face);
          std::array<types::global_dof_index, n_components> first_interior_dofs{};
          neighbor->get_dof_indices(dof_indices);
          for (uint i = 0; i < n_components; ++i)
            first_interior_dofs[i] = dof_indices[i];

          AssertThrow(!is_physical_boundary_face(neighbor, interior_face),
                      ExcMessage("KT boundary stencil requires a second interior cell behind the boundary face."));
          auto next_neighbor = face_neighbor(neighbor, interior_face);
          std::array<types::global_dof_index, n_components> second_interior_dofs{};
          next_neighbor->get_dof_indices(dof_indices);
          for (uint i = 0; i < n_components; ++i)
            second_interior_dofs[i] = dof_indices[i];

          if (boundary_topology.lower_boundary) {
            boundary_topology.x[upper_inner] = neighbor->center();
            boundary_topology.x[upper_outer] = next_neighbor->center();
            boundary_topology.dof_indices[upper_inner] = first_interior_dofs;
            boundary_topology.dof_indices[upper_outer] = second_interior_dofs;
          } else {
            boundary_topology.x[lower_inner] = neighbor->center();
            boundary_topology.x[lower_outer] = next_neighbor->center();
            boundary_topology.dof_indices[lower_inner] = first_interior_dofs;
            boundary_topology.dof_indices[lower_outer] = second_interior_dofs;
          }
        }

        void rebuild_cell_topology_cache()
        {
          FEValues<dim> fe_values(mapping, fe, quadrature, update_quadrature_points | update_JxW_values);
          cell_topology_cache.clear();
          cell_topology_cache.resize(triangulation.n_active_cells());
          boundary_stencil_topologies.clear();

          std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
          // Every cell, not just the owned ones: the stencil reaches two cells out, so an owned
          // cell at a partition boundary needs entries for cells it does not own.
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            auto &cache_entry = cell_topology_cache[cell->active_cell_index()];
            auto &stencil_topology = cache_entry.stencil;
            stencil_topology.neighbors = {};
            stencil_topology.boundary_ids.fill(numbers::invalid_boundary_id);
            stencil_topology.face_centers = {};
            for (auto &neighbor_dofs : stencil_topology.neighbors.dof_indices)
              neighbor_dofs.fill(numbers::invalid_dof_index);
            cache_entry.boundary_stencil_slot.fill(-1);

            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);
            stencil_topology.cell.x = cell->center();
            for (uint i = 0; i < n_components; ++i)
              stencil_topology.cell.dof_indices[i] = dof_indices[i];
            cache_entry.quadrature_points = fe_values.get_quadrature_points();
            cache_entry.jxw.assign(fe_values.get_JxW_values().begin(), fe_values.get_JxW_values().end());
            cache_entry.cell_width = DiFfRG::internal::cell_width(cell);

            for (const auto face_index : cell->face_indices()) {
              const auto face = cell->face(face_index);
              stencil_topology.face_centers[face_index] = face->center();
              if (is_physical_boundary_face(cell, face_index)) {
                stencil_topology.boundary_ids[face_index] = face->boundary_id();
                stencil_topology.neighbors.x[face_index] = face->center();
                cache_entry.boundary_stencil_slot[face_index] = boundary_stencil_topologies.size();
                auto &boundary_reconstruction_topology = boundary_stencil_topologies.emplace_back();
                boundary_reconstruction_topology.tangential_ghost_neighbor_valid.fill(false);
                boundary_reconstruction_topology.corner_tangential_stencil_valid.fill(false);
                for (auto &dofs : boundary_reconstruction_topology.primary.dof_indices)
                  dofs.fill(numbers::invalid_dof_index);
                for (auto &topology : boundary_reconstruction_topology.tangential_ghost_neighbors)
                  for (auto &dofs : topology.dof_indices)
                    dofs.fill(numbers::invalid_dof_index);
                for (auto &corner_stencils : boundary_reconstruction_topology.corner_tangential_stencils)
                  for (auto &topology : corner_stencils)
                    for (auto &dofs : topology.dof_indices)
                      dofs.fill(numbers::invalid_dof_index);
                fill_boundary_topology(boundary_reconstruction_topology.primary, cell, face_index, dof_indices);

                if constexpr (dim == 2) {
                  const unsigned int normal_axis = face_index / 2;
                  const unsigned int tangential_axis = 1U - normal_axis;
                  const unsigned int tangential_minus = 2 * tangential_axis;
                  const unsigned int tangential_plus = tangential_minus + 1;
                  const auto interior_face = GeometryInfo<dim>::opposite_face[face_index];
                  auto normal_neighbor = face_neighbor(cell, interior_face);
                  auto second_normal_neighbor = face_neighbor(normal_neighbor, interior_face);

                  for (const auto tangential_face : {tangential_minus, tangential_plus}) {
                    if (!is_physical_boundary_face(cell, tangential_face)) {
                      const auto tangential_neighbor = face_neighbor(cell, tangential_face);
                      if (is_physical_boundary_face(tangential_neighbor, face_index)) {
                        fill_boundary_topology(
                            boundary_reconstruction_topology.tangential_ghost_neighbors[tangential_face],
                            tangential_neighbor, face_index, dof_indices);
                        boundary_reconstruction_topology.tangential_ghost_neighbor_valid[tangential_face] = true;
                      }
                      continue;
                    }

                    fill_boundary_topology(
                        boundary_reconstruction_topology.corner_tangential_stencils[tangential_face][0], cell,
                        tangential_face, dof_indices);
                    fill_boundary_topology(
                        boundary_reconstruction_topology.corner_tangential_stencils[tangential_face][1],
                        normal_neighbor, tangential_face, dof_indices);
                    fill_boundary_topology(
                        boundary_reconstruction_topology.corner_tangential_stencils[tangential_face][2],
                        second_normal_neighbor, tangential_face, dof_indices);
                    boundary_reconstruction_topology.corner_tangential_stencil_valid[tangential_face] = true;
                  }
                }
                continue;
              }

              const auto neighbor = face_neighbor(cell, face_index);
              neighbor->get_dof_indices(dof_indices);
              stencil_topology.neighbors.x[face_index] = neighbor->center();
              for (uint i = 0; i < n_components; ++i)
                stencil_topology.neighbors.dof_indices[face_index][i] = dof_indices[i];
            }
          }

          // The dependencies read the neighbours' stencils, so every cell's topology has to be in place first.
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            auto &cache_entry = cell_topology_cache[cell->active_cell_index()];
            for (const auto face_index : cell->face_indices())
              build_face_jacobian_dependency_cache(cell, face_index,
                                                   cache_entry.face_jacobian_dependencies[face_index]);
            build_source_jacobian_dependency_cache(cell, cache_entry.source_jacobian_dependencies);
          }
        }

        /// The boundary reconstruction topology of physical boundary face @p face_index of a cell.
        const internal::BoundaryReconstructionStencilTopologyData<dim, n_components> &
        boundary_stencil_topology(const unsigned int cell_index, const unsigned int face_index) const
        {
          const int slot = cell_topology_cache[cell_index].boundary_stencil_slot[face_index];
          Assert(slot >= 0, ExcMessage("Not a physical boundary face."));
          return boundary_stencil_topologies[slot];
        }

        const CellTopologyCacheEntry &get_cell_topology(const Iterator &cell) const
        {
          const auto cell_index = cell->active_cell_index();
          AssertIndexRange(cell_index, cell_topology_cache.size());
          return cell_topology_cache[cell_index];
        }

      protected:
        using Base::discretization;
        using Base::dof_handler;
        using Base::fe;
        using Base::jacobian_times;
        using Base::mapping;
        using Base::model;
        using Base::residual_times;
        using Base::sparsity_pattern_jacobian;
        using Base::timings_jacobian;
        using Base::timings_reinit;
        using Base::timings_residual;

        ReportPort report_port;
        const Triangulation<dim> &triangulation;

        /// The bound on one stacked AD evaluation: /discretization/batched/max_stacked_points.
        size_t max_stacked_points = 0;
        /// The faces the owned cells need (descriptor indices), and the cell's own trace point of every cell face
        /// (-1 if the face is not needed).
        std::vector<unsigned int> trace_faces;
        std::vector<std::array<int, n_faces>> trace_point_of;
        /// The owned cells, colored for the phase-3 scatter; owned cell k has source points k * n_q, ...
        DiFfRG::internal::ColoredCells<dim> cells;
        /// Every active cell by its active_cell_index.
        std::vector<Iterator> active_cells;
        FluxTraces flux_traces;
        DiffusionTraces diffusion_traces;
        SourcePoints source_points;
        /// Per trace point: the advection flux, its value jacobian and the diffusion flux (residual).
        std::vector<TraceFlux> trace_F, trace_D;
        std::vector<TraceJacobian> trace_J;
        /// Per trace point of the current face chunk of face_jacobians(): the advection flux derivatives and the
        /// diffusion flux jacobian.
        std::vector<FluxDerivatives> trace_derivatives;
        std::vector<DiffusionSideJacobian> trace_diffusion_jacobians;
        BatchOutput<dim, NumberType, n_components> source_values;
        std::vector<SourceJacobian> source_jacobians;
        SeedStackWorkspace<SourcePoints, n_components> source_workspace;
        /// The AD workspaces of the trace evaluations (see internal::StackedWorkspace).
        internal::StackedWorkspace<autodiff::Real<1, NumberType>, FluxTraces, n_components> value_seed_workspace;
        internal::FluxDerivativeWorkspace<FluxTraces, n_components> flux_derivative_workspace;
        internal::StackedWorkspace<autodiff::Real<1, NumberType>, DiffusionTraces, n_components>
            diffusion_derivative_workspace;
        /// Per trace face: its residual contribution and its jacobian block, as its u^- cell sees it.
        std::vector<std::array<NumberType, n_components>> face_values;
        std::vector<FullMatrix<NumberType>> face_blocks;
        tbb::enumerable_thread_specific<RowScratch> row_scratch{[this] { return RowScratch(quadrature); }};
        /// The cells' rows for the non-concurrent (PETSc) scatter.
        std::vector<CellRows> row_buffer;

        const QGauss<dim> quadrature;

        std::array<unsigned int, n_components> local_component_of_dof{};
        std::vector<CellTopologyCacheEntry> cell_topology_cache;
        std::vector<internal::BoundaryReconstructionStencilTopologyData<dim, n_components>> boundary_stencil_topologies;
        std::vector<FaceReconstructionDescriptor> face_reconstruction_descriptors;
        SolutionReconstructionCache residual_reconstruction_cache;
        SolutionReconstructionCache jacobian_reconstruction_cache;

        // Warn once per assembler when round-off from a gradient-independent flux baseline
        // exceeds this fraction of the physical flux variation; see
        // probe_diffusion_flux_conditioning(). 1e-9 sits comfortably below the tolerances any
        // fRG flow is run at, so a warning always means real digits are being lost.
        static constexpr double flux_conditioning_warn_threshold = 1e-9;
        static constexpr unsigned int flux_conditioning_max_samples = 512;
        // Set in reinit() so the collective that produces it is never behind a condition.
        double domain_diameter = 0.0;
        // Opt-in ("/discretization/diagnose_flux_conditioning"). The baseline/variation ratio
        // is a sound measure of digits lost in the flux difference, but on its own it cannot
        // tell a harmful case from a harmless one: a model may be baseline-dominated in the
        // deep UV while its diffusion term is still irrelevant to the flow, and pay nothing
        // for it. Enable this when a KT flow is inexplicably slow or stalls at tight
        // tolerances, and read the ratio as "digits available in the diffusion residual".
        const bool diagnose_flux_conditioning;
        mutable bool flux_conditioning_probed = false;
      };
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
