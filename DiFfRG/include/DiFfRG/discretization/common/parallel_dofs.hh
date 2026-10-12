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
  namespace internal
  {
    /// Dof-layout helpers shared by the CG, DG and FV discretizations.
    namespace ParallelDoFs
    {
      /// The ghost set: all dofs, since output, EoM search and FV reconstruction read arbitrary global dofs.
      template <int dim, int spacedim>
      dealii::IndexSet make_ghost_set(const dealii::DoFHandler<dim, spacedim> &dof_handler)
      {
        return dealii::complete_index_set(dof_handler.n_dofs());
      }

      /// The support point of every global dof, on every rank (also for a parallel triangulation).
      template <int dim, int spacedim>
      void build_support_points(const dealii::Mapping<dim, spacedim> &mapping,
                                const dealii::DoFHandler<dim, spacedim> &dof_handler,
                                std::vector<dealii::Point<spacedim>> &support_points)
      {
        const auto n_dofs = dof_handler.n_dofs();
        support_points.assign(n_dofs, dealii::Point<spacedim>());

        const bool is_parallel = (dynamic_cast<const dealii::parallel::TriangulationBase<dim, spacedim> *>(
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
  } // namespace internal
} // namespace DiFfRG
