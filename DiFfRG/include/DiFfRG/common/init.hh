#pragma once

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/threads.hh>

// external libraries
#include <deal.II/base/init_finalize.h>
#include <deal.II/base/mpi.h>

/**
 * @file init.hh
 * @brief DiFfRG::Init: start MPI, PETSc, Kokkos and the CPU thread pool. Construct it first in main().
 *
 * @code
 * int main(int argc, char *argv[])
 * {
 *   DiFfRG::Init init(argc, argv);
 *   const auto config_helper = init.get_configuration_helper();
 *   const DiFfRG::ConfigTree &config = config_helper.get_config();
 *   // ... set up model, discretization, assembler, timestepper and run
 * }
 * @endcode
 */

namespace DiFfRG
{
  /**
   * @brief Set the CPU thread budget from the environment and `/discretization/threads`.
   *
   * For programs that do not construct an Init. Uses the precedence order of ThreadSource; unlike Init it
   * does not divide the automatic budget among several MPI ranks on one node, and prints its warning on
   * every rank.
   *
   * @param config The configuration; only `/discretization/threads` is read.
   */
  void set_thread_limit(const ConfigTree &config);

  /**
   * @brief Starts and owns the libraries DiFfRG runs on: MPI, PETSc, SLEPc, Zoltan, p4est and Kokkos.
   *
   * The first Init of a process does all the work; later ones only store their arguments. The libraries stay
   * up until the process exits, and are shut down by exit handlers, not by ~Init: the Init object may be a
   * temporary. Do not initialize MPI or Kokkos yourself.
   *
   * The first construction
   * - reads the parameter file and command line silently (ConfigurationHelper::probe()) for the keys below,
   * - sets the CPU thread budget, see ThreadSource,
   * - starts MPI and the other libraries,
   * - with several MPI ranks on one node, assigns GPUs round robin: node-local rank \f$r\f$ gets device
   *   \f$r \bmod n_\text{GPU}\f$, and warns if two ranks end up on the same GPU,
   * - starts Kokkos.
   *
   * Keys read here (all optional):
   * - `/discretization/threads` (uint, default 0 = automatic): CPU threads per rank.
   * - `/integration/map_quantum` (double, default 0 = automatic): work-split size of the integrator map
   *   scheduler, see MapScheduler::set_quantum().
   * - `/integration/map_verbose` (bool, default false): log every map schedule.
   */
  class Init
  {
  public:
    /**
     * @brief Start the libraries (first call only) and remember the command line.
     *
     * @param argc Forwarded argc from main().
     * @param argv Forwarded argv from main().
     * @param parameter_file The default parameter file; must be the same file the application reads.
     */
    Init(int argc, char *argv[], const std::string parameter_file = "parameter.json");

    /**
     * @brief Start the libraries without a command line.
     *
     * @param parameter_file The default parameter file.
     */
    Init(const std::string parameter_file = "parameter.json");

    /**
     * @brief Parse the stored command line and parameter file.
     *
     * Builds a new ConfigurationHelper on every call, with all its effects (e.g. `--help` exits).
     */
    const ConfigurationHelper get_configuration_helper() const;

    /// Whether an Init has already started the libraries.
    static bool is_initialized();

  private:
    static bool initialized;

    int argc;
    char **argv;
    std::string parameter_file;
    static dealii::InitFinalize *mpi_initialization;
  };
} // namespace DiFfRG
