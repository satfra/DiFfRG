#pragma once

// standard library
#include <limits>

namespace DiFfRG
{
  /**
   * @brief Implement a simple interface to do all adaptivity tasks, i.e. solution transfer, reinit of dofHandlers, etc.
   *
   * @tparam Assembler The used assembler should implement the methods refinement_indicator and reinit
   */
  template <typename VectorType> class AbstractAdaptor
  {
  public:
    virtual ~AbstractAdaptor() = default;

    /**
     * @brief Check if an adaptation step should be done and tranfer the given solution to the new mesh.
     *
     * @param t Current time; at adapt_dt time distances we perform an adaptation
     * @param sol Current solution
     * @return true if adapation has happened, false otherwise
     */
    virtual bool operator()(const double t, VectorType &sol) = 0;

    /**
     * @brief Force an adaptation and transfer the solution sol to the new mes

     * @param solution to be transferred
     */
    virtual bool adapt(VectorType &solution) = 0;

    /**
     * @brief The time of the last adaptation, or NaN if this adaptor keeps no such schedule.
     *
     * Stored in flow snapshots, so that a restarted flow adapts at the same times as the
     * uninterrupted one would have.
     */
    virtual double last_adaptation_time() const { return std::numeric_limits<double>::quiet_NaN(); }

    /** @brief Restore the adaptation schedule from a snapshot; see last_adaptation_time(). */
    virtual void set_last_adaptation_time(const double /*t*/) {}
  };
} // namespace DiFfRG
