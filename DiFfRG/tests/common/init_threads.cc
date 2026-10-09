#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/threads.hh>

#include <Kokkos_Core.hpp>
#include <deal.II/base/multithread_info.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace
{
  /// Every environment variable the resolver looks at, so a case starts from a known state.
  const std::vector<const char *> thread_env = {"DiFfRG_NUM_THREADS",     "DIFFRG_NUM_THREADS",
                                                "SLURM_CPUS_PER_TASK",    "SLURM_JOB_CPUS_PER_NODE",
                                                "SLURM_CPUS_ON_NODE",     "SLURM_NTASKS_PER_NODE",
                                                "OMP_NUM_THREADS",        "DEAL_II_NUM_THREADS",
                                                "KOKKOS_NUM_THREADS"};

  /// Clears the thread environment for the duration of a test case and restores it afterwards.
  struct CleanThreadEnv {
    std::vector<std::pair<const char *, std::string>> saved;

    CleanThreadEnv()
    {
      for (const char *name : thread_env)
        if (const char *value = std::getenv(name)) {
          saved.emplace_back(name, value);
          ::unsetenv(name);
        }
    }
    ~CleanThreadEnv()
    {
      for (const char *name : thread_env)
        ::unsetenv(name);
      for (const auto &[name, value] : saved)
        ::setenv(name, value.c_str(), 1);
    }
  };

  void set_env(const char *name, const unsigned int value)
  {
    REQUIRE(::setenv(name, std::to_string(value).c_str(), 1) == 0);
  }

  std::string write_config(const std::string &name, const std::string &discretization)
  {
    const std::string filename = name + ".json";
    std::ofstream(filename, std::ofstream::trunc) << "{\"discretization\":" << discretization << "}";
    return filename;
  }
} // namespace

// ------------------------------------------------------------------------------------------------
// Precedence. These exercise the resolver directly: DiFfRG::Init is process-global and can only be
// constructed once, so a precedence order tested through it could only ever check one case.
// ------------------------------------------------------------------------------------------------

TEST_CASE("DiFfRG_NUM_THREADS outranks everything else", "[common][init][threads]")
{
  CleanThreadEnv clean;
  set_env("DiFfRG_NUM_THREADS", 16);
  set_env("SLURM_CPUS_PER_TASK", 8);
  set_env("OMP_NUM_THREADS", 2);
  set_env("DEAL_II_NUM_THREADS", 2);

  const auto r = DiFfRG::resolve_thread_count(4);

  CHECK(r.threads == 16);
  CHECK(r.source == DiFfRG::ThreadSource::diffrg_env);
  CHECK(std::strcmp(r.name, "DiFfRG_NUM_THREADS") == 0);
}

TEST_CASE("DIFFRG_NUM_THREADS is accepted, but the documented spelling wins", "[common][init][threads]")
{
  CleanThreadEnv clean;

  SECTION("on its own")
  {
    set_env("DIFFRG_NUM_THREADS", 7);
    const auto r = DiFfRG::resolve_thread_count(4);
    CHECK(r.threads == 7);
    CHECK(r.source == DiFfRG::ThreadSource::diffrg_env);
  }

  SECTION("alongside the documented spelling")
  {
    set_env("DiFfRG_NUM_THREADS", 5);
    set_env("DIFFRG_NUM_THREADS", 7);
    const auto r = DiFfRG::resolve_thread_count(4);
    CHECK(r.threads == 5);
    CHECK(std::strcmp(r.name, "DiFfRG_NUM_THREADS") == 0);
  }
}

TEST_CASE("A Slurm allocation overrides /discretization/threads in both directions",
          "[common][init][threads][slurm]")
{
  CleanThreadEnv clean;

  SECTION("allocation larger than the configured count")
  {
    set_env("SLURM_CPUS_PER_TASK", 16);
    const auto r = DiFfRG::resolve_thread_count(4);
    CHECK(r.threads == 16);
    CHECK(r.source == DiFfRG::ThreadSource::launcher);
  }

  SECTION("allocation smaller than the configured count")
  {
    set_env("SLURM_CPUS_PER_TASK", 4);
    const auto r = DiFfRG::resolve_thread_count(16);
    CHECK(r.threads == 4);
    CHECK(r.source == DiFfRG::ThreadSource::launcher);
  }
}

TEST_CASE("Per-node Slurm counts are divided among the node's tasks", "[common][init][threads][slurm]")
{
  CleanThreadEnv clean;

  SECTION("SLURM_JOB_CPUS_PER_NODE, compressed list syntax")
  {
    // "72(x2)" means 72 CPUs on each of two nodes -- 72 is this node's count, not 72 * 2.
    REQUIRE(::setenv("SLURM_JOB_CPUS_PER_NODE", "72(x2)", 1) == 0);
    set_env("SLURM_NTASKS_PER_NODE", 4);
    const auto r = DiFfRG::resolve_thread_count(0);
    CHECK(r.threads == 18);
    CHECK(r.source == DiFfRG::ThreadSource::launcher);
  }

  SECTION("SLURM_CPUS_PER_TASK is preferred, being per-task already")
  {
    set_env("SLURM_CPUS_PER_TASK", 9);
    REQUIRE(::setenv("SLURM_JOB_CPUS_PER_NODE", "72", 1) == 0);
    set_env("SLURM_NTASKS_PER_NODE", 4);
    const auto r = DiFfRG::resolve_thread_count(0);
    CHECK(r.threads == 9);
  }
}

TEST_CASE("/discretization/threads outranks OMP_NUM_THREADS and DEAL_II_NUM_THREADS",
          "[common][init][threads]")
{
  CleanThreadEnv clean;
  set_env("OMP_NUM_THREADS", 2);
  set_env("DEAL_II_NUM_THREADS", 3);

  const auto r = DiFfRG::resolve_thread_count(8);

  CHECK(r.threads == 8);
  CHECK(r.source == DiFfRG::ThreadSource::config);
  CHECK(std::strcmp(r.name, "/discretization/threads") == 0);
}

TEST_CASE("OMP_NUM_THREADS applies when nothing better is set, and outranks DEAL_II_NUM_THREADS",
          "[common][init][threads]")
{
  CleanThreadEnv clean;
  set_env("OMP_NUM_THREADS", 6);
  set_env("DEAL_II_NUM_THREADS", 3);

  const auto r = DiFfRG::resolve_thread_count(0);

  // Within tier 4 the order still holds, which is only observable because DEAL_II_NUM_THREADS is
  // removed from the environment before deal.II can take the minimum with it.
  CHECK(r.threads == 6);
  CHECK(r.source == DiFfRG::ThreadSource::other_env);
  CHECK(std::strcmp(r.name, "OMP_NUM_THREADS") == 0);
}

TEST_CASE("Nothing set anywhere falls through to the automatic tier", "[common][init][threads]")
{
  CleanThreadEnv clean;
  const auto r = DiFfRG::resolve_thread_count(0);
  CHECK(r.threads == 0);
  CHECK(r.source == DiFfRG::ThreadSource::automatic);
}

TEST_CASE("A malformed thread count is ignored rather than reinterpreted", "[common][init][threads]")
{
  CleanThreadEnv clean;
  REQUIRE(::setenv("DiFfRG_NUM_THREADS", "8x", 1) == 0);

  const auto r = DiFfRG::resolve_thread_count(4);

  CHECK(r.threads == 4);
  CHECK(r.source == DiFfRG::ThreadSource::config);
}

#ifdef __linux__
TEST_CASE("A narrowed affinity mask outranks the Slurm variables", "[common][init][threads][affinity]")
{
  CleanThreadEnv clean;

  cpu_set_t original;
  CPU_ZERO(&original);
  if (sched_getaffinity(0, sizeof(original), &original) != 0) SKIP("cannot read the affinity mask");
  const unsigned int available = static_cast<unsigned int>(CPU_COUNT(&original));
  if (available < 4) SKIP("needs at least 4 CPUs to narrow the mask meaningfully");

  // Pin to the first two CPUs that are already ours; the mask then describes an allocation
  // strictly smaller than the machine, which is what a launcher's pinning looks like.
  cpu_set_t narrowed;
  CPU_ZERO(&narrowed);
  unsigned int taken = 0;
  for (unsigned int cpu = 0; cpu < CPU_SETSIZE && taken < 2; ++cpu)
    if (CPU_ISSET(cpu, &original)) {
      CPU_SET(cpu, &narrowed);
      ++taken;
    }
  if (sched_setaffinity(0, sizeof(narrowed), &narrowed) != 0) SKIP("cannot narrow the affinity mask");

  set_env("SLURM_CPUS_PER_TASK", 16);
  const auto r = DiFfRG::resolve_thread_count(8);

  CHECK(::sched_setaffinity(0, sizeof(original), &original) == 0);

  CHECK(r.threads == 2);
  CHECK(r.source == DiFfRG::ThreadSource::launcher);
  CHECK(std::strcmp(r.name, "CPU affinity mask") == 0);
}
#endif

// ------------------------------------------------------------------------------------------------
// Installation. One Init per process, so exactly one case here may construct one -- catch_discover_tests
// runs each TEST_CASE as its own ctest test, which is what makes that workable.
// ------------------------------------------------------------------------------------------------

TEST_CASE("Init installs the configured thread count and publishes it", "[common][init][threads][kokkos]")
{
  CleanThreadEnv clean;
  const std::string filename = write_config("init_threads_config", R"({"threads":4})");

  DiFfRG::Init init(filename);

  CHECK(DiFfRG::n_threads() == 4);
  CHECK(DiFfRG::n_threads_source() == DiFfRG::ThreadSource::config);
  // The budget DiFfRG reports and the one deal.II actually installed must never differ.
  CHECK(dealii::MultithreadInfo::n_threads() == DiFfRG::n_threads());
  REQUIRE(Kokkos::is_initialized());

  // A later explicit limit moves both together.
  DiFfRG::set_thread_limit(2);
  CHECK(DiFfRG::n_threads() == 2);
  CHECK(dealii::MultithreadInfo::n_threads() == 2);

  std::filesystem::remove(filename);
}

TEST_CASE("A CLI override reaches the resolver before the libraries come up",
          "[common][init][threads][kokkos][cli]")
{
  CleanThreadEnv clean;
  const std::string filename = write_config("init_threads_cli", R"({"threads":2})");
  char program[] = "init_threads";
  char parameter_flag[] = "-p";
  char threads_flag[] = "-si";
  char threads_override[] = "/discretization/threads=4";
  // argv[argc] must be a null pointer, as for main(): MPI_Init (Open MPI 5) walks argv up to it.
  char *argv[] = {program, parameter_flag, const_cast<char *>(filename.c_str()), threads_flag, threads_override,
                  nullptr};

  DiFfRG::Init init(5, argv);

  CHECK(DiFfRG::n_threads() == 4);
  REQUIRE(Kokkos::is_initialized());
  std::filesystem::remove(filename);
}

TEST_CASE("DEAL_II_NUM_THREADS does not cap a larger OMP_NUM_THREADS either",
          "[common][init][threads][environment]")
{
  CleanThreadEnv clean;
  const std::string filename = write_config("init_threads_omp_env", R"({"threads":0})");
  set_env("OMP_NUM_THREADS", 6);
  set_env("DEAL_II_NUM_THREADS", 3);

  DiFfRG::Init init(filename);

  // Both are tier 4, so the order within the tier decides -- which is only observable because
  // DEAL_II_NUM_THREADS is removed rather than left for deal.II to take the minimum with.
  CHECK(DiFfRG::n_threads() == 6);
  CHECK(dealii::MultithreadInfo::n_threads() == 6);

  std::filesystem::remove(filename);
}

TEST_CASE("DEAL_II_NUM_THREADS no longer caps a higher-priority setting",
          "[common][init][threads][environment]")
{
  CleanThreadEnv clean;
  const std::string filename = write_config("init_threads_dealii_env", R"({"threads":4})");
  set_env("DEAL_II_NUM_THREADS", 2);

  DiFfRG::Init init(filename);

  // deal.II takes the minimum with DEAL_II_NUM_THREADS on every set_thread_limit() call, so
  // honouring /discretization/threads above it requires removing the variable. Without that, this
  // reports 2.
  CHECK(DiFfRG::n_threads() == 4);
  CHECK(dealii::MultithreadInfo::n_threads() == 4);
  CHECK(std::getenv("DEAL_II_NUM_THREADS") == nullptr);

  std::filesystem::remove(filename);
}
