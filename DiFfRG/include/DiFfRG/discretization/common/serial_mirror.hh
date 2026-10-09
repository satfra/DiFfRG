#pragma once

// external libraries
#include <boost/signals2/connection.hpp>

#include <deal.II/distributed/tria_base.h>
#include <deal.II/grid/tria.h>

// standard library
#include <map>
#include <memory>
#include <mutex>

namespace DiFfRG
{
  /**
   * @brief A process-local, serial mirror of a (possibly partitioned) triangulation.
   *
   * Returns @p source itself when it is already serial, so a serial build allocates nothing and
   * behaves exactly as before.
   *
   * Why this exists: at the replicated-mesh rung every rank holds the whole mesh and (through
   * SolutionView) the whole solution, so anything that only needs to *read* the mesh can and should
   * do so without communicating. But deal.II decides collectiveness from the triangulation's type,
   * not from what the caller intends:
   *
   *  * DoFHandler::distribute_dofs() on a parallel triangulation is collective.
   *  * A loop over the locally owned cells (internal::ColoredCells) visits only the calling rank's
   *    cells.
   *  * DataOut::add_data_vector() routes every vector type through a ghosted
   *    LinearAlgebra::distributed::BlockVector built on dof_handler.get_mpi_communicator().
   *
   * Each of those is a hang or a silently truncated result when it runs inside a rank-0-only region
   * such as OutputSession's contributor. Handing that code a serial mirror makes all three local,
   * complete and identical on every rank.
   *
   * The mirror is cached and refreshed IN PLACE, and the stable address is load-bearing:
   * dealii::DataOut latches onto the triangulation of the first DoFHandler it is handed and never
   * lets go -- DataOut_DoFData::clear() resets `dofs` but leaves `triangulation` alone, and
   * add_data_vector only re-reads it while it is still null. A fresh mirror per call therefore
   * leaves the output pointing at a destroyed mesh, which surfaces much later as
   * ReferenceCell::get_default_linear_mapping() throwing ExcNotImplemented out of build_patches.
   *
   * Refreshed when the source mesh changes: the any_change signal covers refinement, coarsening and
   * clearing, and the size comparison covers a source we never managed to connect to.
   */
  template <int dim> const dealii::Triangulation<dim> &serial_mirror(const dealii::Triangulation<dim> &source)
  {
    if (dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(&source) == nullptr) return source;

    struct Mirror {
      std::unique_ptr<dealii::Triangulation<dim>> triangulation;
      boost::signals2::scoped_connection connection;
      bool stale = true;
    };

    static std::mutex mirrors_mutex;
    static std::map<const dealii::Triangulation<dim> *, Mirror> mirrors;

    std::scoped_lock lock(mirrors_mutex);
    auto &mirror = mirrors[&source];
    // A dead connection means the triangulation this entry was built for is gone and something else
    // now occupies its address, so the cached copy describes a different mesh entirely. Destroying a
    // triangulation disconnects its slots, which makes this a reliable test.
    if (!mirror.triangulation || !mirror.connection.connected()) {
      if (!mirror.triangulation) mirror.triangulation = std::make_unique<dealii::Triangulation<dim>>();
      mirror.connection = source.signals.any_change.connect([&source]() {
        std::scoped_lock relock(mirrors_mutex);
        const auto it = mirrors.find(&source);
        if (it != mirrors.end()) it->second.stale = true;
      });
      mirror.stale = true;
    }

    auto &copy = *mirror.triangulation;
    const bool size_mismatch = copy.n_active_cells() != source.n_active_cells() ||
                               copy.n_levels() != source.n_levels() || copy.n_vertices() != source.n_vertices();
    if (mirror.stale || size_mismatch) {
      copy.clear();
      copy.copy_triangulation(source);
      mirror.stale = false;
    }
    return copy;
  }
} // namespace DiFfRG
