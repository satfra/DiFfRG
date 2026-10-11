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
  namespace internal
  {
    /**
     * @brief A serial copy of a parallel triangulation, or @p source itself if it is serial.
     *
     * Lets code that runs on one rank only (output, potential reconstruction) use deal.II functions that are collective
     * on a parallel triangulation. The copy is cached, updated in place when the source changes, and keeps its address.
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
  } // namespace internal
} // namespace DiFfRG
