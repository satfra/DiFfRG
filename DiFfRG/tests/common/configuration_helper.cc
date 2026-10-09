#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <deal.II/base/multithread_info.h>
#include <fstream>

#ifdef __linux__
#include <sched.h>
#endif

#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/output_path.hh>
#include <DiFfRG/discretization/data/snapshot.hh>

#include <filesystem>
#include <iostream>
#include <sstream>

using namespace DiFfRG;

TEST_CASE("Test configuration helper", "[config][common]")
{
  SECTION("File io")
  {
    // Generate some random parameter names and values
    json::value jv = {{"a", 1},
                      {"b", 2},
                      {"c", 3},
                      {"d", 4},
                      {"e", 5},
                      {"f", 6},
                      {"g", 7},
                      {"h", 8},
                      {"i", 9},
                      {"j", true},
                      {"output", {{"folder", "./"}, {"name", "output"}, {"verbosity", 0}}}};

    // Write jv to a file
    const std::string filename = "test_config_helper.json";
    std::ofstream file(filename);
    file << jv;
    file.close();

    // Use a ConfigurationHelper to read the file
    int argc = 3;
    char *argv[] = {(char *)"test", (char *)"-p", (char *)filename.c_str()};
    ConfigurationHelper config(argc, argv);

    // Check that the values are read correctly
    REQUIRE(config.get_config().get_int("/a") == 1);
    REQUIRE(config.get_config().get_int("/b") == 2);
    REQUIRE(config.get_config().get_int("/c") == 3);
    REQUIRE(config.get_config().get_int("/d") == 4);
    REQUIRE(config.get_config().get_int("/e") == 5);
    REQUIRE(config.get_config().get_int("/f") == 6);
    REQUIRE(config.get_config().get_int("/g") == 7);
    REQUIRE(config.get_config().get_int("/h") == 8);
    REQUIRE(config.get_config().get_int("/i") == 9);
    REQUIRE(config.get_config().get_bool("/j") == true);
  }
}

TEST_CASE("The snapshot flags create their keys", "[config][common][snapshot]")
{
  // The parameter file does not mention /timestepping/snapshots: unlike -sd & co., these flags must
  // create the keys rather than fail on them.
  const std::string filename = "test_config_helper_snapshots.json";
  {
    std::ofstream file(filename);
    file << json::value({{"timestepping", {{"final_time", 1.}}}, {"output", {{"verbosity", 0}}}});
  }
  char *argv[] = {(char *)"test",
                  (char *)"-p",
                  (char *)filename.c_str(),
                  (char *)"--snapshots-k",
                  (char *)"2,0.5",
                  (char *)"--snapshots-t",
                  (char *)"1.5",
                  (char *)"--stop-after-last-snapshot"};
  ConfigurationHelper helper(8, argv);
  const auto &config = helper.get_config();

  REQUIRE(config.get_double("/timestepping/final_time") == 1.);
  REQUIRE(config.get_bool("/timestepping/snapshots/stop_after_last"));
  const json::value root = config;
  REQUIRE(root.at_pointer("/timestepping/snapshots/k") == json::value(json::array{2., 0.5}));
  REQUIRE(root.at_pointer("/timestepping/snapshots/t") == json::value(json::array{1.5}));
}

TEST_CASE("A restart takes its configuration from the snapshot", "[config][common][snapshot]")
{
  const std::string snapshot_file = "test_config_helper_seed_snapshot.h5";
  std::filesystem::remove(snapshot_file);
  SnapshotData seed;
  seed.config_json = json::serialize(json::value(
      {{"physical", {{"T", 0.05}, {"Lambda", 1.}}},
       {"timestepping", {{"final_time", 2.}, {"snapshots", {{"k", json::array{0.5}}, {"stop_after_last", true}}}}},
       {"restart", {{"file", "older_snapshot.h5"}}},
       {"output", {{"name", "seed"}, {"verbosity", 0}}}}));
  write_snapshot(snapshot_file, seed);

  // Not read: a restart must not pick up anything from it.
  const std::string parameter_file = "test_config_helper_restart.json";
  std::ofstream(parameter_file) << json::value({{"physical", {{"T", 1.}, {"Lambda", 3.}}}});

  const auto parse = [&](std::vector<std::string> extra) {
    std::vector<std::string> args{"test", "--restart", snapshot_file};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<char *> argv;
    for (auto &arg : args)
      argv.push_back(arg.data());
    return ConfigurationHelper::probe(int(argv.size()), argv.data(), parameter_file);
  };

  SECTION("The snapshot's values, CLI overrides on top")
  {
    const auto config = parse({"-sd", "/physical/T=0.1", "-ss", "/output/name=restarted"});
    REQUIRE(config.get_double("/physical/T") == 0.1);
    REQUIRE(config.get_double("/physical/Lambda") == 1.);
    REQUIRE(config.get_double("/timestepping/final_time") == 2.);
    REQUIRE(config.get_string("/output/name") == "restarted");
    REQUIRE(config.get_string("/restart/file") == snapshot_file);
    // The seed run's snapshot schedule is not inherited.
    REQUIRE_FALSE(config.contains("/timestepping/snapshots"));
  }
  SECTION("A restart may request snapshots of its own")
  {
    const json::value config = parse({"--snapshots-k", "0.2"});
    REQUIRE(config.at_pointer("/timestepping/snapshots/k") == json::value(json::array{0.2}));
    REQUIRE_FALSE(
        config.as_object().at("timestepping").as_object().at("snapshots").as_object().contains("stop_after_last"));
  }
  SECTION("Every override is printed at the start")
  {
    std::ostringstream captured;
    auto *const previous = std::cout.rdbuf(captured.rdbuf());
    std::vector<std::string> args{"test", "--restart", snapshot_file, "-sd", "/physical/T=0.1"};
    std::vector<char *> argv;
    for (auto &arg : args)
      argv.push_back(arg.data());
    ConfigurationHelper helper(int(argv.size()), argv.data(), parameter_file);
    std::cout.rdbuf(previous);
    INFO(captured.str());
    REQUIRE(captured.str().find("Overridden on the command line (1)") != std::string::npos);
    REQUIRE(captured.str().find("/physical/T: 0.05 -> 0.1") != std::string::npos);
  }
  SECTION("Invalid combinations are errors")
  {
    // probe() returns an empty tree where the application's own parse would stop with an error.
    REQUIRE_FALSE(parse({"-sd", "/physical/not_in_the_snapshot=1"}).contains("/physical"));
    REQUIRE_FALSE(parse({"-p", parameter_file}).contains("/physical"));
  }
  SECTION("A restart from the run's own snapshot continues it, with its schedule")
  {
    const auto folder = std::filesystem::absolute("test_config_helper_continue");
    std::filesystem::remove_all(folder);
    std::filesystem::create_directories(folder);
    const auto own_snapshot = (folder / "seed_snapshot_000.h5").string();
    SnapshotData own;
    own.config_json = json::serialize(
        json::value({{"timestepping", {{"final_time", 2.}, {"snapshots", {{"t", json::array{0.5, 1.}}}}}},
                     {"output", {{"name", "seed"}, {"folder", folder.string()}, {"verbosity", 0}}}}));
    write_snapshot(own_snapshot, own);
    const auto restart = [&](std::vector<std::string> extra) {
      std::vector<std::string> args{"test", "--restart", own_snapshot};
      args.insert(args.end(), extra.begin(), extra.end());
      std::vector<char *> argv;
      for (auto &arg : args)
        argv.push_back(arg.data());
      return json::value(ConfigurationHelper::probe(int(argv.size()), argv.data(), parameter_file));
    };

    REQUIRE(restart({}).at_pointer("/timestepping/snapshots/t") == json::value(json::array{0.5, 1.}));
    // Into another output it is a new run, without the schedule ...
    REQUIRE_FALSE(
        restart({"-ss", "/output/name=other"}).as_object().at("timestepping").as_object().contains("snapshots"));
    // ... and snapshot flags replace the schedule as a whole.
    const auto replaced = restart({"--snapshots-k", "0.2"});
    REQUIRE(replaced.at_pointer("/timestepping/snapshots/k") == json::value(json::array{0.2}));
    REQUIRE_FALSE(replaced.as_object().at("timestepping").as_object().at("snapshots").as_object().contains("t"));
    std::filesystem::remove_all(folder);
  }
  SECTION("/restart in a parameter file is an error")
  {
    const std::string file = "test_config_helper_restart_key.json";
    std::ofstream(file) << json::value({{"restart", {{"file", snapshot_file}}}});
    char *argv[] = {(char *)"test"};
    REQUIRE_FALSE(ConfigurationHelper::probe(1, argv, file).contains("/restart"));
  }
}

TEST_CASE("Test configuration helper with TOML", "[config][common][toml]")
{
  const std::string document = R"TOML(
[physical]
T = 0.1
Lambda = 0.65

[output]
folder = "./"
name = "output"
verbosity = 0
)TOML";

  SECTION("A TOML file given with -p is read, and CLI overrides still apply")
  {
    const std::string filename = "test_config_helper.toml";
    std::ofstream(filename, std::ofstream::trunc) << document;

    int argc = 7;
    char *argv[] = {(char *)"test",
                    (char *)"-p",
                    (char *)filename.c_str(),
                    (char *)"-sd",
                    (char *)"/physical/T=0.5",
                    (char *)"-ss",
                    (char *)"/output/name=renamed"};
    ConfigurationHelper config(argc, argv);

    CHECK(config.get_config().get_double("/physical/Lambda") == Catch::Approx(0.65));
    CHECK(config.get_config().get_double("/physical/T") == Catch::Approx(0.5));
    CHECK(config.get_config().get_string("/output/name") == "renamed");
    // get_json() is the historical spelling of the same accessor.
    CHECK(config.get_json().get_double("/physical/Lambda") == Catch::Approx(0.65));

    std::filesystem::remove(filename);
  }

  SECTION("A missing default parameter file falls back to the TOML file of the same name")
  {
    const std::string stem = "test_config_helper_fallback";
    std::ofstream(stem + ".toml", std::ofstream::trunc) << document;
    std::filesystem::remove(stem + ".json");

    int argc = 1;
    char *argv[] = {(char *)"test"};
    ConfigurationHelper config(argc, argv, stem + ".json");

    CHECK(config.get_parameter_file() == stem + ".toml");
    CHECK(config.get_config().get_double("/physical/Lambda") == Catch::Approx(0.65));

    std::filesystem::remove(stem + ".toml");
  }
}

TEST_CASE("probe() reads the configuration without diagnostics or exiting", "[config][common][threads]")
{
  const std::string document = R"JSON({"discretization": {"threads": 3, "fe_order": 2, "overintegration": 7}})JSON";

  SECTION("A -p file and CLI overrides are honoured just like the normal parse")
  {
    const std::string filename = "test_probe.json";
    std::ofstream(filename, std::ofstream::trunc) << document;

    int argc = 7;
    char *argv[] = {(char *)"test",
                    (char *)"-p",
                    (char *)filename.c_str(),
                    (char *)"-si",
                    (char *)"/discretization/threads=5",
                    (char *)"-si",
                    (char *)"/discretization/fe_order=1"};
    const ConfigTree config = ConfigurationHelper::probe(argc, argv);

    CHECK(config.get_uint("/discretization/threads", 0) == 5);
    CHECK(config.get_uint("/discretization/fe_order", 0) == 1);
    CHECK(config.get_uint("/discretization/overintegration", 0) == 7);

    std::filesystem::remove(filename);
  }

  SECTION("An absent parameter file yields an empty tree rather than terminating")
  {
    int argc = 1;
    char *argv[] = {(char *)"test"};
    const ConfigTree config = ConfigurationHelper::probe(argc, argv, "test_probe_does_not_exist.json");

    CHECK_FALSE(config.contains("/discretization/threads"));
    CHECK(config.get_uint("/discretization/threads", 0) == 0);
  }

  SECTION("--help and --generate-parameter-file do not terminate a probe")
  {
    int argc = 3;
    char *argv[] = {(char *)"test", (char *)"--help", (char *)"--generate-parameter-file"};
    const ConfigTree config = ConfigurationHelper::probe(argc, argv, "test_probe_does_not_exist.json");

    CHECK_FALSE(config.contains("/discretization/threads"));
    CHECK_FALSE(std::filesystem::exists("test_probe_does_not_exist.json"));
  }
}

TEST_CASE("contains() distinguishes a missing key from a present one", "[config][common]")
{
  const ConfigTree config(json::value{{"discretization", {{"threads", 4}}}});

  CHECK(config.contains("/discretization/threads"));
  CHECK_FALSE(config.contains("/discretization/overintegration"));
  CHECK_FALSE(config.contains("/nonexistent/section/key"));
  // A defaulted getter must agree with contains().
  CHECK(config.get_uint("/discretization/threads", 99) == 4);
  CHECK(config.get_uint("/discretization/overintegration", 99) == 99);
}

namespace
{
  /// CPUs this process may run on: what the automatic thread budget is derived from.
  unsigned int available_cpus()
  {
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) return static_cast<unsigned int>(CPU_COUNT(&mask));
#endif
    return dealii::MultithreadInfo::n_cores();
  }
} // namespace

TEST_CASE("set_thread_limit caps the process thread count", "[config][common][threads]")
{
  const unsigned int original = dealii::MultithreadInfo::n_threads();

  set_thread_limit(2);
  CHECK(dealii::MultithreadInfo::n_threads() == 2);

  // A tree without the key means "decide from the CPUs this process actually has", which is the
  // affinity mask -- the whole machine unless a launcher or taskset narrowed it. Notably it must
  // recompute rather than inherit the cap set just above.
  set_thread_limit(ConfigTree(json::value{{"discretization", {{"overintegration", 8}}}}));
  CHECK(dealii::MultithreadInfo::n_threads() == available_cpus());

  set_thread_limit(ConfigTree(json::value{{"discretization", {{"threads", 3}}}}));
  CHECK(dealii::MultithreadInfo::n_threads() == 3);

  set_thread_limit(original);
}

TEST_CASE("Legacy configuration output getters forward to OutputPath", "[config][common][migration]")
{
  const auto root = std::filesystem::absolute("configuration-helper-output").lexically_normal();
  const ConfigTree config(
      json::value{{"output", {{"folder", root.string()}, {"name", "run"}, {"field_directory", "fields"}}}});
  const ConfigurationHelper config_helper(config);
  const OutputPath output_path(config);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  CHECK(config_helper.get_log_file() == output_path.run_file(".log").filename().string());
  CHECK(config_helper.get_output_name() == output_path.run_name());
  CHECK(config_helper.get_output_folder() == make_folder(output_path.field_directory().generic_string()));
  CHECK(config_helper.get_top_folder() == make_folder(output_path.root().generic_string()));
#pragma GCC diagnostic pop
}
