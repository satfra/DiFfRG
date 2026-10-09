#pragma once

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/threads.hh>

// external libraries
#include <deal.II/base/init_finalize.h>
#include <deal.II/base/mpi.h>

namespace DiFfRG
{
  /**
   * @brief Limit the number of CPU threads from a configuration tree.
   *
   * Applies the same precedence order as DiFfRG::Init -- environment, launcher allocation, then
   * /discretization/threads -- so an embedder that skips Init still honours a cluster allocation.
   */
  void set_thread_limit(const ConfigTree &config);

  class Init
  {
  public:
    /**
     * Initialize deal.II's supported libraries and the process-global Kokkos runtime.
     *
     * The CPU thread budget is resolved once, here, following the precedence order documented at
     * ThreadSource; DiFfRG warns loudly when several sources disagree. Kokkos' host backend is
     * Serial (see the KOKKOS_THREADS build option), so the only CPU thread pool is TBB's and the
     * budget applies to it alone.
     */
    Init(int argc, char *argv[], const std::string parameter_file = "parameter.json");
    Init(const std::string parameter_file = "parameter.json");

    const ConfigurationHelper get_configuration_helper() const;

    static bool is_initialized();

  private:
    static bool initialized;

    int argc;
    char **argv;
    std::string parameter_file;
    static dealii::InitFinalize *mpi_initialization;
  };
} // namespace DiFfRG
