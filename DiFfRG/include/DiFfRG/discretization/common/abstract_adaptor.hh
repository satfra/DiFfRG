#pragma once

// standard library
#include <limits>
#include <stdexcept>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief Interface of mesh adaptors: decide when to adapt the mesh, adapt it, and transfer the solution.
   *
   * Implementations: HAdaptivity (refinement and coarsening of cells) and NoAdaptivity (fixed mesh).
   *
   * @tparam VectorType Vector type of the spatial solution.
   */
  template <typename VectorType> class AbstractAdaptor
  {
  public:
    virtual ~AbstractAdaptor() = default;

    /**
     * @brief Adapt the mesh if one is due at time @p t, and transfer @p sol to the new mesh.
     *
     * @param t The current RG time.
     * @param sol The current solution; resized and interpolated if the mesh changed.
     * @return true if the mesh changed.
     */
    virtual bool operator()(const double t, VectorType &sol) = 0;

    /**
     * @brief As operator()(t, sol), but also transfers the vectors in @p also_transfer (states on the same mesh as
     * @p sol, e.g. a stepper's earlier checkpoints) to the new mesh.
     *
     * The default implementation can transfer nothing else: it throws if an adaptation happened and
     * @p also_transfer is not empty.
     */
    virtual bool adapt_at(const double t, VectorType &sol, const std::vector<VectorType *> &also_transfer)
    {
      const bool adapted = (*this)(t, sol);
      if (adapted && !also_transfer.empty())
        throw std::logic_error("This mesh adaptor cannot transfer additional vectors; implement adapt_at().");
      return adapted;
    }

    /**
     * @brief Adapt the mesh now, regardless of the schedule, and transfer @p solution to the new mesh.
     *
     * @return true if the mesh changed.
     */
    virtual bool adapt(VectorType &solution) = 0;

    /**
     * @brief The time of the last adaptation, or NaN if this adaptor keeps no such schedule.
     *
     * Stored in flow snapshots, so that a restarted flow adapts at the same times as an uninterrupted one.
     */
    virtual double last_adaptation_time() const { return std::numeric_limits<double>::quiet_NaN(); }

    /// Restore the adaptation schedule from a snapshot; see last_adaptation_time().
    virtual void set_last_adaptation_time(const double /*t*/) {}
  };
} // namespace DiFfRG
