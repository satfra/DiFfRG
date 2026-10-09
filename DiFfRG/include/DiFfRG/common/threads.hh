#pragma once

namespace DiFfRG
{
  /**
   * @brief Where the CPU thread budget came from, in order of precedence (highest first).
   *
   * DiFfRG runs its CPU work on a single TBB pool, so there is exactly one number to resolve, and
   * several places that may want to set it. They are ranked rather than combined: a launcher knows
   * more about the machine than a parameter file does, and an explicit environment variable is the
   * user overruling both.
   */
  enum class ThreadSource {
    /// DiFfRG_NUM_THREADS (or DIFFRG_NUM_THREADS).
    diffrg_env,
    /// The CPU allocation this process was given: its affinity mask, or SLURM_* if it has none.
    launcher,
    /// /discretization/threads.
    config,
    /// OMP_NUM_THREADS, DEAL_II_NUM_THREADS, KOKKOS_NUM_THREADS.
    other_env,
    /// Nothing was set: the machine, divided among the node-local ranks that share its CPUs.
    automatic
  };

  /// The name of a thread-count source, as it appears in the precedence warning.
  const char *to_string(const ThreadSource source);

  /**
   * @brief The CPU thread budget this process resolved.
   *
   * This is the number every DiFfRG component sizes itself against: the host/device split of the
   * map scheduler, and the TBB arena itself. Resolved once by
   * DiFfRG::Init, or by set_thread_limit() for embedders that never construct an Init.
   *
   * Deliberately not a synonym for dealii::MultithreadInfo::n_threads(). That is a mutable static
   * which every set_thread_limit() call rewrites -- silently taking the minimum with
   * DEAL_II_NUM_THREADS as it goes -- so the value read back out of deal.II is not necessarily the
   * value DiFfRG resolved. The two are kept in agreement here by construction: this returns what
   * deal.II actually installed, after the precedence rules have been applied.
   *
   * Before resolution it reports TBB's live concurrency, so an embedder that skips Init still gets
   * a usable number rather than a zero.
   */
  unsigned int n_threads();

  /// Which rule produced n_threads(). ThreadSource::automatic before anything has been resolved.
  ThreadSource n_threads_source();

  /// What the precedence rules picked, and which setting it came from.
  struct ThreadResolution {
    /// 0 when nothing was set anywhere, i.e. the automatic tier applies.
    unsigned int threads;
    ThreadSource source;
    /// The winning setting's name, e.g. "SLURM_CPUS_PER_TASK". Never null.
    const char *name;
  };

  /**
   * @brief Apply the precedence order to the environment and one configured thread count.
   *
   * Pure: it reads the environment and reports what would win, without installing anything and
   * without touching the environment. DiFfRG::Init calls this and then acts on the result; it is
   * public so the precedence order can be tested directly, rather than through a process-global
   * Init that can only be constructed once.
   *
   * @param configured_threads /discretization/threads, or 0 if unset.
   */
  ThreadResolution resolve_thread_count(const unsigned int configured_threads);

  /**
   * @brief Limit the number of CPU threads this process may use.
   *
   * Sets deal.II's thread limit, which installs the process-wide tbb::global_control capping
   * every TBB pipeline in deal.II and DiFfRG, and publishes the result through n_threads().
   * A value of 0 means "all available cores".
   *
   * @note This is the unconditional form: it does not consult the environment. Applications go
   * through DiFfRG::Init, which applies the precedence order described at ThreadSource. This
   * exists for tests and for embedders that drive the library without Init.
   */
  void set_thread_limit(const unsigned int threads);
} // namespace DiFfRG
