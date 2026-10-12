#pragma once

/**
 * @file threads.hh
 * @brief The CPU thread budget of a DiFfRG process.
 *
 * All CPU work runs in one TBB thread pool. Its size is taken from the first of the following that is set
 * (see ThreadSource):
 * 1. the environment variable `DiFfRG_NUM_THREADS` (or `DIFFRG_NUM_THREADS`),
 * 2. the CPUs the launcher gave this process: its CPU affinity mask if it is smaller than the machine,
 *    otherwise `SLURM_CPUS_PER_TASK`, otherwise `SLURM_JOB_CPUS_PER_NODE` or `SLURM_CPUS_ON_NODE` divided by
 *    `SLURM_NTASKS_PER_NODE`,
 * 3. the parameter `/discretization/threads` (0 means unset),
 * 4. `OMP_NUM_THREADS`, `DEAL_II_NUM_THREADS` or `KOKKOS_NUM_THREADS`,
 * 5. otherwise all CPUs, shared among the MPI ranks on the node.
 *
 * DiFfRG::Init prints a warning when several of these are set to different values. Note that tier 2 beats
 * the parameter file: e.g. `mpirun` binding each rank to one core gives one thread per rank.
 */

namespace DiFfRG
{
  /**
   * @brief Where the CPU thread budget came from, in order of precedence (highest first).
   *
   * See threads.hh for the full rules.
   */
  enum class ThreadSource {
    /// `DiFfRG_NUM_THREADS` (or `DIFFRG_NUM_THREADS`).
    diffrg_env,
    /// The CPU allocation this process was given: its affinity mask, or `SLURM_*` if it has none.
    launcher,
    /// `/discretization/threads`, or an explicit set_thread_limit(unsigned int).
    config,
    /// `OMP_NUM_THREADS`, `DEAL_II_NUM_THREADS`, `KOKKOS_NUM_THREADS`.
    other_env,
    /// Nothing was set: the machine, divided among the node-local ranks that share its CPUs.
    automatic
  };

  /// The name of a thread-count source, as it appears in log messages.
  const char *to_string(const ThreadSource source);

  /**
   * @brief The number of CPU threads this process uses.
   *
   * Set by DiFfRG::Init, and changed by every later set_thread_limit() call. Before either, TBB's current
   * concurrency limit.
   */
  unsigned int n_threads();

  /// Which rule produced n_threads(). ThreadSource::automatic before anything has been set.
  ThreadSource n_threads_source();

  /// What the precedence rules picked, and which setting it came from.
  struct ThreadResolution {
    /// 0 when nothing was set anywhere, i.e. the automatic tier applies.
    unsigned int threads;
    /// The tier of the winning setting.
    ThreadSource source;
    /// The winning setting's name, e.g. "SLURM_CPUS_PER_TASK". Never null.
    const char *name;
  };

  /**
   * @brief Apply the precedence order to the environment and one configured thread count.
   *
   * Only reports what would win; changes nothing. Exposed for tests.
   *
   * @param configured_threads `/discretization/threads`, or 0 if unset.
   */
  ThreadResolution resolve_thread_count(const unsigned int configured_threads);

  /**
   * @brief Set the number of CPU threads this process may use.
   *
   * Sets deal.II's thread limit, which caps every TBB pool in deal.II and DiFfRG, and updates n_threads().
   * 0 means all available cores. Skips the precedence order of threads.hh, but deal.II still caps the value
   * at `DEAL_II_NUM_THREADS` if that is set.
   *
   * @param threads The thread count.
   */
  void set_thread_limit(const unsigned int threads);
} // namespace DiFfRG
