#pragma once

// external libraries
#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/fe/mapping.h>

// DiFfRG
#include <DiFfRG/common/mpi.hh>

// std
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief DoF-layout helpers shared by the CG, DG and FV discretizations.
   *
   * These exist because the answer to "which dofs does this rank hold?" differs between a
   * serial and a partitioned triangulation, and every discretization needs the same answer.
   */
  namespace ParallelDoFs
  {
    /**
     * @brief The ghost (locally relevant) set used by the *replicated-mesh* policy: everything.
     *
     * This is deliberately NOT DoFTools::extract_locally_relevant_dofs. The KT reconstruction
     * stencil reaches three cells deep normal to a boundary and reads corner/diagonal cells that
     * are not face neighbours at all (see FV/assembler/reconstruction_cache.hh), and the EoM
     * search and the output path read arbitrary global indices. A one-ring relevant set is not
     * wide enough for any of them, and being too narrow is a silent wrong answer rather than an
     * error.
     *
     * Ghosting everything costs exactly today's memory footprint, which is the whole reason the
     * replicated-mesh rung is cheap. Narrowing this to extract_locally_relevant_dofs is the
     * defining step of the fully-distributed rung, and every call site that reads a global index
     * has to be revisited at the same time.
     */
    template <int dim, int spacedim>
    dealii::IndexSet make_ghost_set(const dealii::DoFHandler<dim, spacedim> &dof_handler)
    {
      return dealii::complete_index_set(dof_handler.n_dofs());
    }

    /**
     * @brief Fill a dense, mesh-wide support point array on every rank.
     *
     * Why this is not just DoFTools::map_dofs_to_support_points: the std::vector overload of that
     * function *refuses to run* on a DoFHandler built on a parallel::TriangulationBase -- and
     * parallel::shared::Triangulation is one -- because its precondition is an array sized to the
     * global dof count. Calling it on a partitioned mesh is an error, not a silent degradation.
     *
     * The std::map overload does work in parallel, but returns only the locally relevant dofs. The
     * union of those sets over all ranks is the whole mesh, so a max-reduction over a
     * lowest()-initialized array reconstructs the dense global array exactly. Support points are
     * pure geometry -- every rank that holds a given dof computes bit-identical coordinates for it
     * -- so the reduction is a gather, not an approximation, and the result does not depend on the
     * rank count.
     *
     * @param support_points resized to dof_handler.n_dofs() and fully populated on every rank.
     */
    template <int dim, int spacedim>
    void build_support_points(const dealii::Mapping<dim, spacedim> &mapping,
                              const dealii::DoFHandler<dim, spacedim> &dof_handler,
                              std::vector<dealii::Point<spacedim>> &support_points)
    {
      const auto n_dofs = dof_handler.n_dofs();
      support_points.assign(n_dofs, dealii::Point<spacedim>());

      const bool is_parallel =
          (dynamic_cast<const dealii::parallel::TriangulationBase<dim, spacedim> *>(
               &dof_handler.get_triangulation()) != nullptr);

      if (!is_parallel) {
        dealii::DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);
        return;
      }

      // Flat [n_dofs * spacedim] buffer so the reduction is a single collective. lowest() marks
      // "this rank knows nothing about this dof"; max() then picks the real value from whichever
      // rank does.
      constexpr double unset = std::numeric_limits<double>::lowest();
      std::vector<double> flat(static_cast<std::size_t>(n_dofs) * spacedim, unset);

      const auto local_points = dealii::DoFTools::map_dofs_to_support_points(mapping, dof_handler);
      for (const auto &[dof, point] : local_points)
        for (uint d = 0; d < spacedim; ++d)
          flat[static_cast<std::size_t>(dof) * spacedim + d] = point[d];

      MPI::max_reduce(dof_handler.get_mpi_communicator(), flat.data(), static_cast<int>(flat.size()));

      for (dealii::types::global_dof_index i = 0; i < n_dofs; ++i)
        for (uint d = 0; d < spacedim; ++d) {
          const double value = flat[static_cast<std::size_t>(i) * spacedim + d];
          if (value == unset)
            throw std::runtime_error("ParallelDoFs::build_support_points: dof " + std::to_string(i) +
                                     " was not locally relevant on any rank. The ghost set is "
                                     "narrower than the dof distribution, which would silently "
                                     "corrupt every support-point lookup.");
          support_points[i][d] = value;
        }
    }
  } // namespace ParallelDoFs
} // namespace DiFfRG
