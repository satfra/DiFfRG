#include <DiFfRG/common/init.hh>

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/stream_filter.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>

// standard libraries
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// external libraries
#include <deal.II/base/kokkos.h>
#include <deal.II/base/multithread_info.h>
#include <tbb/tbb.h>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef __linux__
#include <sched.h>
#endif

namespace DiFfRG
{
  namespace
  {
    /// Length of the PCI bus id string CUDA returns, e.g. "0000:65:00.0".
    constexpr int bus_id_len = 16;

#if defined(DiFfRG_KERNEL_ARCH) && defined(DiFfRG_BUNDLE_ARCH)
    // Kokkos determines "the" compiled architecture by launching a probe kernel that reads
    // __CUDA_ARCH__. That kernel lives in libkokkoscore, so on a dependency bundle built for an
    // older architecture than the device it reports the *bundle's*, and warns that kernels will run
    // below their potential. For DiFfRG that is misleading twice over: the statement covers only a
    // few hundred Kokkos and deal.II plumbing kernels, and everything carrying the physics is
    // compiled natively for this GPU (see DiFfRG_CUDA_ARCH). Init() suppresses the line and prints
    // an accurate one instead.
    //
    // Kokkos::show_warnings() would also silence it, but takes every other Kokkos warning with it.
    constexpr std::string_view kokkos_arch_warning =
        "Kokkos::Cuda::initialize WARNING: running kernels compiled for compute capability";
#endif

    // ------------------------------------------------------------------------------------------
    // The resolved CPU thread budget
    // ------------------------------------------------------------------------------------------

    /// 0 until something resolves it; see DiFfRG::n_threads().
    unsigned int resolved_threads = 0;
    ThreadSource resolved_source = ThreadSource::automatic;

    /**
     * @brief Install a thread limit and record it as the process' budget.
     *
     * deal.II owns the tbb::global_control that actually caps the arena, so it has to be told; but
     * it also takes the minimum with DEAL_II_NUM_THREADS while doing so. Reading the number back
     * out of deal.II afterwards is what keeps DiFfRG::n_threads() and
     * dealii::MultithreadInfo::n_threads() from ever disagreeing.
     */
    void install_thread_limit(const unsigned int threads, const ThreadSource source)
    {
      dealii::MultithreadInfo::set_thread_limit(threads == 0 ? dealii::numbers::invalid_unsigned_int : threads);
      resolved_threads = dealii::MultithreadInfo::n_threads();
      resolved_source = source;
    }

    // ------------------------------------------------------------------------------------------
    // Reading thread counts out of the environment
    // ------------------------------------------------------------------------------------------

    /// The leading integer of @p text, if it starts with one. Returns nothing for 0 or garbage.
    std::optional<unsigned int> parse_leading_count(const char *text)
    {
      size_t i = 0;
      while (text[i] != '\0' && std::isspace(static_cast<unsigned char>(text[i])))
        ++i;
      size_t j = i;
      while (text[j] != '\0' && std::isdigit(static_cast<unsigned char>(text[j])))
        ++j;
      if (j == i) return std::nullopt;
      unsigned long value = 0;
      try {
        value = std::stoul(std::string(text + i, j - i));
      } catch (...) {
        return std::nullopt;
      }
      // An implausibly large count is far more likely to be a misparse than an intention.
      if (value == 0 || value > 100000) return std::nullopt;
      return static_cast<unsigned int>(value);
    }

    /**
     * @brief A thread count set by the user, which must be exactly a positive number.
     *
     * Anything else is a typo worth reporting rather than silently reinterpreting.
     */
    std::optional<unsigned int> env_count_strict(const char *name)
    {
      const char *text = std::getenv(name);
      if (text == nullptr || *text == '\0') return std::nullopt;
      const auto parsed = parse_leading_count(text);
      // Reject trailing junk: "8" is a count, "8x" is a mistake.
      if (parsed) {
        const std::string digits = std::to_string(*parsed);
        std::string trimmed(text);
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));
        trimmed.erase(trimmed.find_last_not_of(" \t") + 1);
        if (trimmed == digits) return parsed;
      }
      std::cerr << "WARNING: " << name << "='" << text << "' is not a positive thread count; ignoring it." << std::endl;
      return std::nullopt;
    }

    /**
     * @brief A thread count set by Slurm, which uses compressed list syntax.
     *
     * SLURM_JOB_CPUS_PER_NODE is e.g. "72", "72(x2)" or "36,72" -- the first entry describes this
     * node, and the rest is not ours to interpret.
     */
    std::optional<unsigned int> env_count_leading(const char *name)
    {
      const char *text = std::getenv(name);
      if (text == nullptr || *text == '\0') return std::nullopt;
      return parse_leading_count(text);
    }

    /// CPUs in this process' affinity mask, or 0 where that cannot be asked.
    unsigned int affinity_cpu_count()
    {
#ifdef __linux__
      cpu_set_t mask;
      CPU_ZERO(&mask);
      if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return 0;
      return static_cast<unsigned int>(CPU_COUNT(&mask));
#else
      return 0;
#endif
    }

    // ------------------------------------------------------------------------------------------
    // Precedence
    // ------------------------------------------------------------------------------------------

    /// One place that had an opinion about the thread count. Names are string literals.
    struct Candidate {
      const char *name;
      unsigned int value;
      ThreadSource tier;
    };

    struct Resolution {
      /// False when nothing was set anywhere, i.e. the automatic tier applies.
      bool resolved = false;
      unsigned int threads = 0;
      ThreadSource source = ThreadSource::automatic;
      const char *winner = "";
      std::vector<Candidate> seen;
      /// DEAL_II_NUM_THREADS was removed from the environment; see resolve_threads().
      bool dealii_env_discarded = false;
    };

    /**
     * @brief What this process was allocated by whoever launched it.
     *
     * The affinity mask is authoritative and is checked first: it is the only thing that describes
     * the CPUs this process may actually run on, and it covers taskset, cgroups and every launcher
     * rather than just Slurm. It is meaningful only when it is a *strict* subset of the machine --
     * an unrestricted mask is what an ordinary workstation looks like, and must not be mistaken for
     * an allocation. Note that n_cores() is std::thread::hardware_concurrency(), which ignores the
     * mask, so this comparison is between the mask and the whole machine, as intended.
     *
     * Slurm's variables are the fallback for the case where a job did not pin its tasks.
     */
    std::optional<Candidate> launcher_allocation()
    {
      const unsigned int cpus = affinity_cpu_count();
      if (cpus > 0 && cpus < dealii::MultithreadInfo::n_cores())
        return Candidate{"CPU affinity mask", cpus, ThreadSource::launcher};

      if (const auto v = env_count_leading("SLURM_CPUS_PER_TASK"))
        return Candidate{"SLURM_CPUS_PER_TASK", *v, ThreadSource::launcher};

      // The per-node counts have to be shared out among the tasks placed on the node.
      const unsigned int tasks = std::max(1u, env_count_leading("SLURM_NTASKS_PER_NODE").value_or(1u));
      for (const char *name : {"SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_ON_NODE"})
        if (const auto v = env_count_leading(name))
          return Candidate{name, std::max(1u, *v / tasks), ThreadSource::launcher};

      return std::nullopt;
    }

    /**
     * @brief Apply the precedence order documented at ThreadSource.
     *
     * Everything here is knowable before MPI_Init: it is environment and parameter file only. The
     * automatic tier is not, because it needs the node-local rank count, so this reports
     * `resolved == false` and leaves that case to automatic_budget().
     */
    Resolution resolve_threads(const unsigned int configured_threads)
    {
      Resolution r;

      // Tier 1: an explicit DiFfRG setting. Both spellings are accepted; the documented one wins.
      if (const auto v = env_count_strict("DiFfRG_NUM_THREADS"))
        r.seen.push_back({"DiFfRG_NUM_THREADS", *v, ThreadSource::diffrg_env});
      if (const auto v = env_count_strict("DIFFRG_NUM_THREADS"))
        r.seen.push_back({"DIFFRG_NUM_THREADS", *v, ThreadSource::diffrg_env});

      // Tier 2: the CPUs this process was actually given.
      if (const auto c = launcher_allocation()) r.seen.push_back(*c);

      // Tier 3: the parameter file.
      if (configured_threads > 0)
        r.seen.push_back({"/discretization/threads", configured_threads, ThreadSource::config});

      // Tier 4: thread counts meant for somebody else, which we honour only if nothing better said
      // anything. DEAL_II_NUM_THREADS is read here rather than left to deal.II -- see
      // discard_dealii_env_override().
      const auto dealii_env = env_count_strict("DEAL_II_NUM_THREADS");
      if (const auto v = env_count_strict("OMP_NUM_THREADS"))
        r.seen.push_back({"OMP_NUM_THREADS", *v, ThreadSource::other_env});
      if (dealii_env) r.seen.push_back({"DEAL_II_NUM_THREADS", *dealii_env, ThreadSource::other_env});
      if (const auto v = env_count_strict("KOKKOS_NUM_THREADS"))
        r.seen.push_back({"KOKKOS_NUM_THREADS", *v, ThreadSource::other_env});

      // min_element returns the *first* smallest, so within a tier the order above breaks ties.
      const auto winner = std::min_element(r.seen.begin(), r.seen.end(),
                                           [](const Candidate &a, const Candidate &b) { return a.tier < b.tier; });
      if (winner == r.seen.end()) return r;

      r.resolved = true;
      r.threads = winner->value;
      r.source = winner->tier;
      r.winner = winner->name;

      return r;
    }

    /**
     * @brief Take DEAL_II_NUM_THREADS out of the environment where it would undercut the winner.
     *
     * dealii::MultithreadInfo::set_thread_limit() takes the minimum with DEAL_II_NUM_THREADS on
     * every call, so unless the variable is itself the winner it would quietly override whatever
     * won from below and make the whole precedence order a lie -- including inside tier 4, where a
     * larger OMP_NUM_THREADS ought to win. Where that would happen, remove the variable and say so;
     * it is listed in the conflict warning either way.
     *
     * Kept out of resolve_threads() so that resolution stays free of side effects and can be
     * called from a test as often as it likes.
     */
    void discard_dealii_env_override(Resolution &r)
    {
      if (!r.resolved || std::strcmp(r.winner, "DEAL_II_NUM_THREADS") == 0) return;
      const char *text = std::getenv("DEAL_II_NUM_THREADS");
      if (text == nullptr || *text == '\0') return;
      const auto value = parse_leading_count(text);
      if (!value || *value >= r.threads) return;
      ::unsetenv("DEAL_II_NUM_THREADS");
      r.dealii_env_discarded = true;
    }

    /// Print every setting that had an opinion, once, when they do not all agree.
    void warn_about_conflicts(const Resolution &r, const unsigned int threads)
    {
      const bool disagree =
          std::any_of(r.seen.begin(), r.seen.end(), [&](const Candidate &c) { return c.value != threads; });
      if (!disagree) return;

      // stderr on purpose: timestepper progress is stdout-only, and a cluster launcher may well
      // swallow one of the two streams.
      std::cerr << "\n"
                << "================================ WARNING ================================\n"
                << " Several thread-count settings disagree:\n";
      for (const auto &c : r.seen) {
        std::cerr << "   " << std::left << std::setw(26) << c.name << "= " << c.value;
        if (std::strcmp(c.name, r.winner) == 0)
          std::cerr << "   <-- used";
        else if (r.dealii_env_discarded && std::strcmp(c.name, "DEAL_II_NUM_THREADS") == 0)
          std::cerr << "   (removed from the environment: deal.II would have capped the above)";
        std::cerr << "\n";
      }
      std::cerr << " Running with " << threads << " CPU threads, from " << r.winner << ".\n"
                << " Precedence: DiFfRG_NUM_THREADS > launcher allocation (CPU affinity, then\n"
                << " SLURM_*) > /discretization/threads > OMP_NUM_THREADS / DEAL_II_NUM_THREADS\n"
                << " > automatic.\n"
                << "=========================================================================\n"
                << std::endl;
    }

    // ------------------------------------------------------------------------------------------
    // The automatic tier
    // ------------------------------------------------------------------------------------------

    /**
     * @brief This rank's CPU budget when nothing was configured, and why.
     */
    struct CpuBudget {
      unsigned int threads;
      /// Node-local ranks competing for the same CPUs, this rank included. 1 == exclusive.
      unsigned int sharing;
      /// Whether the launcher handed this rank its own CPUs.
      bool pinned;
    };

    /**
     * @brief Work out how many CPU threads this rank may start when nobody said.
     *
     * The naive version of this -- take the machine's core count and divide by the number of
     * node-local ranks -- is wrong on exactly the systems that matter, and wrong in both directions:
     *
     *  - `std::thread::hardware_concurrency()`, which is what deal.II's `MultithreadInfo::n_cores()`
     *    calls, **ignores the CPU affinity mask**. Measured on this machine: pinned to 8 of 32 CPUs
     *    with taskset, it still reports 32. So a rank pinned to its own cores does not know it.
     *  - Conversely, when a launcher *has* pinned each rank (Slurm `--cpus-per-task`), TBB's
     *    `default_concurrency()` does see the mask (8 in the same test), so the thread limit is
     *    already per-rank -- and dividing it again by the node-local rank count halves it a second
     *    time. On a 128-core node split into 2 pinned ranks that yields 32 threads each, not 64.
     *
     * So the divisor is not "how many ranks are on this node" but "how many node-local ranks are
     * competing for *my* CPUs". That is answered by intersecting the affinity masks, the same way
     * GPU sharing is answered by comparing bus ids rather than counting devices.
     */
    CpuBudget automatic_budget([[maybe_unused]] const unsigned int local_size)
    {
#ifdef __linux__
      cpu_set_t mask;
      CPU_ZERO(&mask);
      if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
        return CpuBudget{std::max(1u, dealii::MultithreadInfo::n_cores() / std::max(1u, local_size)), local_size,
                         false};

      const unsigned int my_cpus = static_cast<unsigned int>(CPU_COUNT(&mask));

      unsigned int sharing = 1;
      if (local_size > 1) {
        // Gather every node-local rank's mask and count how many overlap mine. Disjoint masks mean
        // the launcher already partitioned the node; identical masks mean the ranks share it.
        MPI_Comm node_comm = MPI::split_shared(MPI_COMM_WORLD);
        const size_t bytes = sizeof(cpu_set_t);
        std::vector<char> all(bytes * local_size, '\0');
        std::vector<int> counts(local_size, static_cast<int>(bytes));
        std::vector<int> displs(local_size);
        for (unsigned int i = 0; i < local_size; ++i)
          displs[i] = static_cast<int>(i * bytes);
        MPI::allgatherv_bytes(node_comm, &mask, bytes, all.data(), counts, displs);
        MPI::free_comm(node_comm);

        const char *mine = reinterpret_cast<const char *>(&mask);
        sharing = 0;
        for (unsigned int p = 0; p < local_size; ++p) {
          const char *other = all.data() + p * bytes;
          bool overlaps = false;
          for (size_t b = 0; b < bytes && !overlaps; ++b)
            overlaps = (mine[b] & other[b]) != 0;
          if (overlaps) ++sharing;
        }
        if (sharing == 0) sharing = 1; // cannot happen: a mask always overlaps itself
      }

      // The mask is already a subset of the machine, so no further capping is needed -- and none is
      // wanted: capping against the limit currently in force would make repeated calls ratchet the
      // budget down instead of recomputing it. An embedder's own tbb::global_control still applies
      // regardless of what deal.II is told, because TBB takes the minimum over all live instances.
      return CpuBudget{std::max(1u, my_cpus / sharing), sharing, sharing == 1 && local_size > 1};
#else
      // No portable way to read an affinity mask; fall back to an even split of the node.
      return CpuBudget{std::max(1u, dealii::MultithreadInfo::n_cores() / std::max(1u, local_size)), local_size, false};
#endif
    }

    /**
     * @brief Give this rank its own GPU, and warn loudly if it has to share one.
     *
     * Without this every rank on a node lands on device 0, which is the single biggest thing
     * standing between a multi-rank launch and actual multi-GPU execution.
     *
     * Sharing is detected by comparing the devices' **PCI bus ids**, not by counting. Under Slurm's
     * `--gpus-per-task=1` every rank sees exactly one device and `local_rank % n_devices == 0` is
     * the correct answer; a count-based check would report sharing on every well-configured
     * cluster. It is a warning rather than an error because a shared-GPU run is still correct, just
     * slower -- and a scaling study has to be able to measure that case.
     *
     * @return the device id this rank should use, or -1 for "no choice to make".
     */
    int select_device([[maybe_unused]] const unsigned int local_rank, [[maybe_unused]] const unsigned int local_size)
    {
#ifdef KOKKOS_ENABLE_CUDA
      int n_devices = 0;
      if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices <= 0) return -1;

      const int device = static_cast<int>(local_rank % static_cast<unsigned int>(n_devices));
      if (local_size <= 1) return device;

      char bus_id[bus_id_len] = {};
      if (cudaDeviceGetPCIBusId(bus_id, bus_id_len, device) != cudaSuccess) return device;

      // Gather every node-local rank's bus id and look for duplicates. This is the only way to see
      // through CUDA_VISIBLE_DEVICES, which renumbers devices per process.
      MPI_Comm node_comm = MPI::split_shared(MPI_COMM_WORLD);
      std::vector<char> all_ids(static_cast<size_t>(local_size) * bus_id_len, '\0');
      std::vector<int> counts(local_size, bus_id_len);
      std::vector<int> displs(local_size);
      for (unsigned int i = 0; i < local_size; ++i)
        displs[i] = static_cast<int>(i * bus_id_len);
      MPI::allgatherv_bytes(node_comm, bus_id, bus_id_len, all_ids.data(), counts, displs);

      std::vector<unsigned int> sharing;
      for (unsigned int p = 0; p < local_size; ++p)
        if (std::strncmp(all_ids.data() + p * bus_id_len, bus_id, bus_id_len) == 0) sharing.push_back(p);

      if (sharing.size() > 1) {
        // stderr on every affected rank: timestepper progress is stdout-only and a cluster launcher
        // may well swallow one of the two streams.
        std::string ranks;
        for (size_t i = 0; i < sharing.size(); ++i)
          ranks += (i ? ", " : "") + std::to_string(sharing[i]);
        std::cerr << "\n"
                  << "================================ WARNING ================================\n"
                  << " GPU " << bus_id << " is shared by " << sharing.size() << " node-local ranks (" << ranks << ").\n"
                  << " The run is still correct, but those ranks time-slice the device and each\n"
                  << " costs several hundred MB of extra context, so it will NOT scale.\n"
                  << " Launch with one GPU per rank (e.g. Slurm --gpus-per-task=1), or set\n"
                  << " CUDA_VISIBLE_DEVICES per rank.\n"
                  << "=========================================================================\n"
                  << std::endl;
      }
      MPI::free_comm(node_comm);
      return device;
#else
      return -1;
#endif
    }
  } // namespace

  const char *to_string(const ThreadSource source)
  {
    switch (source) {
    case ThreadSource::diffrg_env:
      return "DiFfRG_NUM_THREADS";
    case ThreadSource::launcher:
      return "the launcher's CPU allocation";
    case ThreadSource::config:
      return "/discretization/threads";
    case ThreadSource::other_env:
      return "OMP_NUM_THREADS / DEAL_II_NUM_THREADS";
    case ThreadSource::automatic:
      return "the available CPUs";
    }
    return "an unknown source";
  }

  ThreadResolution resolve_thread_count(const unsigned int configured_threads)
  {
    const Resolution r = resolve_threads(configured_threads);
    return ThreadResolution{r.resolved ? r.threads : 0u, r.source, r.winner};
  }

  unsigned int n_threads()
  {
    if (resolved_threads > 0) return resolved_threads;
    // Nothing has resolved a budget yet, so report what TBB would actually run with.
    return static_cast<unsigned int>(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism));
  }

  ThreadSource n_threads_source() { return resolved_source; }

  void set_thread_limit(const unsigned int threads) { install_thread_limit(threads, ThreadSource::config); }

  void set_thread_limit(const ConfigTree &config)
  {
    Resolution r = resolve_threads(config.get_uint("/discretization/threads", 0));
    if (r.resolved) {
      discard_dealii_env_override(r);
      install_thread_limit(r.threads, r.source);
      warn_about_conflicts(r, n_threads());
    } else {
      const CpuBudget budget = automatic_budget(1);
      install_thread_limit(budget.threads, ThreadSource::automatic);
    }
  }

  bool Init::initialized = false;

  dealii::InitFinalize *Init::mpi_initialization = nullptr;

  Init::Init(int argc, char *argv[], const std::string parameter_file)
      : argc(argc), argv(argv), parameter_file(parameter_file)
  {
    if (!initialized) {
      // The thread count has to be known *before* the libraries come up: dealii::InitFinalize
      // forwards it to MultithreadInfo::set_thread_limit(), which installs the process-wide
      // tbb::global_control that caps every TBB pipeline in deal.II and DiFfRG. probe() therefore
      // takes an early, silent look at the parameter file; the application's own
      // ConfigurationHelper does the authoritative parse.
      const ConfigTree config = ConfigurationHelper::probe(argc, argv, parameter_file);

      if (config.contains("/discretization/kokkos_threads"))
        std::cerr << "WARNING: '/discretization/kokkos_threads' has been removed and is ignored.\n"
                     "         Kokkos' host backend is Serial: DiFfRG's CPU work runs on TBB, and a second\n"
                     "         thread pool only contended with it. '/discretization/threads' now sizes the\n"
                     "         one pool there is. Please delete the entry from your parameter file.\n"
                  << std::endl;

      if (config.contains("/discretization/mesh_workers") || config.contains("/discretization/batch_size"))
        std::cerr << "WARNING: '/discretization/mesh_workers' and '/discretization/batch_size' are ignored.\n"
                     "         They sized the MeshWorker pipeline of the assemblers, which no longer use one.\n"
                     "         Please delete them from your parameter file.\n"
                  << std::endl;

      // Everything except the automatic tier is knowable here, before MPI is up. Resolving now is
      // what lets the answer be handed straight to dealii::InitFinalize; it also has to happen
      // before the first set_thread_limit() call, because that is where a stale DEAL_II_NUM_THREADS
      // would otherwise silently cap us.
      Resolution resolution = resolve_threads(config.get_uint("/discretization/threads", 0));
      discard_dealii_env_override(resolution);
      const unsigned int max_num_threads =
          resolution.resolved ? resolution.threads
                              : tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);

      // We initialize every library that deal.II's MPI_InitFinalize would, *except* Kokkos, and then
      // hand the Kokkos lifecycle to deal.II via ensure_kokkos_initialized() below. The reason is a
      // teardown-ordering bug:
      //
      // MPI_InitFinalize initializes Kokkos by calling Kokkos::initialize() directly, which leaves
      // dealii::internal::dealii_initialized_kokkos == false. With that flag false, deal.II's
      // GrowingVectorMemory installs a Kokkos finalize-hook (release_unused_memory) that dereferences
      // a function-local-static memory pool. Because we finalize Kokkos from a std::atexit handler,
      // that hook runs *after* the pool static has already been destroyed during normal static
      // teardown -> dereference of freed memory -> intermittent segfault on program exit.
      //
      // ensure_kokkos_initialized() instead sets dealii_initialized_kokkos = true (so the dangerous
      // finalize-hook is never installed) and registers Kokkos::finalize() itself via std::atexit.
      // The vector-memory pool is then emptied by its own (~Pool) destructor while Kokkos is still
      // alive, and Kokkos is finalized afterwards.
      mpi_initialization = new dealii::InitFinalize(
          argc, argv,
          dealii::InitializeLibrary::MPI | dealii::InitializeLibrary::SLEPc | dealii::InitializeLibrary::PETSc |
              dealii::InitializeLibrary::Zoltan | dealii::InitializeLibrary::P4EST,
          max_num_threads);

      std::atexit([]() {
        if (initialized) {
          // Finalizes MPI/PETSc/... (but not Kokkos, which deal.II finalizes via its own atexit hook
          // registered by ensure_kokkos_initialized()).
          delete Init::mpi_initialization;
          initialized = false;
        }
      });

      // From here on MPI_Init has run, so the node-local rank is finally knowable -- which is why
      // the automatic tier could not be folded into max_num_threads above.
      MPI_Comm node_comm = MPI::split_shared(MPI_COMM_WORLD);
      const unsigned int local_rank = MPI::rank(node_comm);
      const unsigned int local_size = MPI::size(node_comm);
      MPI::free_comm(node_comm);

      CpuBudget budget{max_num_threads, 1, false};
      if (resolution.resolved) {
        install_thread_limit(resolution.threads, resolution.source);
      } else {
        // Without a per-rank cap, N ranks each start a full-width TBB arena on the same cores and
        // spend their time fighting each other. automatic_budget() works out the right cap from the
        // affinity masks rather than by dividing a machine-wide count -- see its comment for why the
        // obvious version is wrong in both directions.
        budget = automatic_budget(local_size);
        install_thread_limit(budget.threads, ThreadSource::automatic);
      }

      if (MPI::rank(MPI_COMM_WORLD) == 0) {
        warn_about_conflicts(resolution, n_threads());
        if (local_size > 1)
          std::cerr << "DiFfRG: " << local_size << " ranks per node, " << n_threads() << " CPU threads each ("
                    << (resolution.resolved ? std::string(resolution.winner) + ", applied per rank"
                        : budget.pinned     ? std::string("the launcher pinned each rank to its own CPUs")
                                            : std::to_string(budget.sharing) + " ranks share one CPU set")
                    << ")." << std::endl;
      }

      const int device = select_device(local_rank, local_size);

      // Initialize Kokkos *after* registering the teardown above so that Kokkos::finalize() (also
      // registered via std::atexit, here) runs before the MPI finalization, matching the order
      // deal.II's own ~MPI_InitFinalize uses.
      //
      // With more than one rank we cannot simply call ensure_kokkos_initialized(): it offers no way
      // to pick a device, and every rank would land on device 0. We therefore do exactly what it
      // would have done -- including setting dealii_initialized_kokkos and registering the atexit
      // hook, BOTH of which live inside its `if (!Kokkos::is_initialized())` branch and would
      // otherwise be skipped. Getting that wrong reintroduces the teardown segfault described
      // above: GrowingVectorMemory installs a Kokkos finalize-hook into an already-destroyed
      // static, and nothing ever finalizes Kokkos.
      //
      // num_threads configures only Kokkos' host execution backend, which is Serial unless the
      // build enabled KOKKOS_THREADS; it never restricts CUDA kernel parallelism. It is passed
      // anyway so that a build which does enable that backend still respects the budget.
#if defined(DiFfRG_KERNEL_ARCH) && defined(DiFfRG_BUNDLE_ARCH)
      // Drop Kokkos' architecture warning while it initializes; see kokkos_arch_warning above for why.
      const bool arch_differs = std::string(DiFfRG_KERNEL_ARCH) != DiFfRG_BUNDLE_ARCH;
      std::optional<ScopedLineFilter> arch_warning_filter;
      if (arch_differs) arch_warning_filter.emplace(std::cerr, kokkos_arch_warning);
#endif

      if (device >= 0 && local_size > 1 && !Kokkos::is_initialized() && !Kokkos::is_finalized()) {
        dealii::internal::dealii_initialized_kokkos = true;
        Kokkos::initialize(
            Kokkos::InitializationSettings().set_device_id(device).set_num_threads(static_cast<int>(n_threads())));
        std::atexit([]() { Kokkos::finalize(); });
      } else {
        dealii::internal::ensure_kokkos_initialized();
      }

#if defined(DiFfRG_KERNEL_ARCH) && defined(DiFfRG_BUNDLE_ARCH)
      arch_warning_filter.reset();
      // Say once what the suppressed warning was trying to say, correctly.
      if (arch_differs && device >= 0 && MPI::rank(MPI_COMM_WORLD) == 0)
        std::cerr << "DiFfRG: kernels compiled for " << DiFfRG_KERNEL_ARCH << "; the bundled Kokkos and deal.II are "
                  << "built for " << DiFfRG_BUNDLE_ARCH << " and are the only ones this GPU has to JIT-compile."
                  << std::endl;
#endif

      // The scheduler defaults are fine for most runs; expose them anyway, since the split width is
      // the one knob a scaling study may want to turn.
      MapScheduler::instance().set_quantum(config.get_double("/integration/map_quantum", 0.));
      if (config.get_bool("/integration/map_verbose", false)) MapScheduler::instance().set_verbose(true);

      initialized = true;
    }
  }

  Init::Init(const std::string parameter_file) : Init(0, nullptr, parameter_file) {}

  const ConfigurationHelper Init::get_configuration_helper() const
  {
    ConfigurationHelper configuration_helper(argc, argv, parameter_file);
    return configuration_helper;
  }

  bool Init::is_initialized() { return initialized; }
} // namespace DiFfRG
