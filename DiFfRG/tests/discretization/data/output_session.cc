#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/discretization/data/output_session.hh>

#include <deal.II/lac/vector.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <spdlog/spdlog.h>

namespace
{
  // Apple's libc++ ships no std::jthread; this test only needs the
  // join-on-destruction behavior, so use a minimal stand-in.
  struct joining_thread {
    template <typename... Args>
    explicit joining_thread(Args &&...args) : t(std::forward<Args>(args)...)
    {
    }
    joining_thread(joining_thread &&) = default;
    ~joining_thread()
    {
      if (t.joinable()) t.join();
    }
    void join() { t.join(); }
    std::thread t;
  };

  DiFfRG::Config::OutputSettings output_settings()
  {
    DiFfRG::Config::OutputSettings settings;
    settings.Lambda = 1.;
    settings.write_vtk = false;
    settings.write_hdf5 = false;
    return settings;
  }

  DiFfRG::ConfigTree output_json(const std::filesystem::path &path)
  {
    return DiFfRG::json::value(
        {{"physical", {{"Lambda", 1.0}}},
         {"discretization", {{"output_subdivisions", 1}}},
         {"output",
          {{"folder", path.string()}, {"name", "run"}, {"vtk", false}, {"hdf5", false}, {"max_pending_frames", 2}}}});
  }

  std::string read_file(const std::filesystem::path &path)
  {
    std::ifstream stream(path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  }

  std::size_t count_occurrences(const std::string &text, const std::string &needle)
  {
    std::size_t count = 0;
    for (std::size_t position = 0; (position = text.find(needle, position)) != std::string::npos;
         position += needle.size())
      ++count;
    return count;
  }
} // namespace

TEST_CASE("OutputSession rejects an incomplete table frame", "[output][session]")
{
  auto path = DiFfRG::OutputPath::temporary();

  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());
    output.write_frame(0.0, [](auto &frame) {
      auto table = frame.table("data.csv");
      table.value("a", 1.0);
      table.value("b", 2.0);
    });

    CHECK_THROWS_WITH(output.write_frame(1.0, [](auto &frame) { frame.table("data.csv").value("a", 3.0); }),
                      Catch::Matchers::ContainsSubstring("incomplete"));
    CHECK_THROWS_WITH(output.write_frame(2.0, [](auto &) {}), Catch::Matchers::ContainsSubstring("incomplete"));
  }
}

TEST_CASE("DiagnosticPort serializes concurrent records", "[output][diagnostics][thread-safe]")
{
  auto path = DiFfRG::OutputPath::temporary();

  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());
    const auto diagnostics = output.diagnostic_port();

    std::vector<joining_thread> writers;
    for (int i = 0; i < 16; ++i)
      writers.emplace_back([diagnostics, i]() { diagnostics.scalar("eigenvalues.csv", "lambda_max", i, i * 0.5); });
  }

  std::ifstream stream(path.root() / "output_eigenvalues.csv");
  REQUIRE(stream.good());
  std::size_t line_count = 0;
  for (std::string line; std::getline(stream, line);)
    ++line_count;
  CHECK(line_count == 17);
}

TEST_CASE("DiagnosticPort is fail-stop after a malformed record", "[output][diagnostics][schema]")
{
  auto path = DiFfRG::OutputPath::temporary();

  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());
    const auto diagnostics = output.diagnostic_port();
    diagnostics.record("coefficients.csv", 0.0, {{"diffusion", 1.0}, {"viscosity", 2.0}});
    CHECK_THROWS_WITH(diagnostics.scalar("coefficients.csv", "diffusion", 1.0, 3.0),
                      Catch::Matchers::ContainsSubstring("incomplete"));
    CHECK_THROWS_WITH(diagnostics.scalar("coefficients.csv", "diffusion", 2.0, 4.0),
                      Catch::Matchers::ContainsSubstring("incomplete"));
  }
}

TEST_CASE("OutputSession rejects path traversal and use after finish", "[output][session][layout]")
{
  auto path = DiFfRG::OutputPath::temporary();

  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());
    CHECK_FALSE(spdlog::get("run"));
    CHECK_THROWS_WITH(output.write_frame(0.0, [](auto &frame) { frame.table("../escape.csv"); }),
                      Catch::Matchers::ContainsSubstring("unsafe"));
    CHECK_THROWS_WITH(output.finish(), Catch::Matchers::ContainsSubstring("unsafe"));
  }
  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());
    output.finish();
    CHECK_THROWS_WITH(output.write_frame(1.0, [](auto &) {}),
                      Catch::Matchers::ContainsSubstring("already been finished"));
  }
}

TEST_CASE("OutputSession drain keeps the session writable until finish", "[output][session][lifecycle]")
{
  auto path = DiFfRG::OutputPath::temporary();
  DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());

  output.write_frame(0.0, [](auto &frame) { frame.table("data.csv").value("value", 1.0); });
  REQUIRE_NOTHROW(output.drain());

  output.write_frame(1.0, [](auto &frame) { frame.table("data.csv").value("value", 2.0); });
  REQUIRE_NOTHROW(output.drain());

  output.finish();
  CHECK_THROWS_WITH(output.write_frame(2.0, [](auto &) {}),
                    Catch::Matchers::ContainsSubstring("already been finished"));
}

TEST_CASE("Run log can be flushed while the session is alive", "[output][session][log]")
{
  auto path = DiFfRG::OutputPath::temporary();
  const auto log_file = path.run_file(".log");

  DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());
  output.report_port().info("periodic flush marker");
  output.report_port().flush();

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  bool found = false;
  while (!found && std::chrono::steady_clock::now() < deadline) {
    std::ifstream stream(log_file);
    std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    found = content.find("periodic flush marker") != std::string::npos;
    if (!found) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  CHECK(found);
}

TEST_CASE("Default ReportPort logs to the console without creating a file", "[output][reporter][default]")
{
  DiFfRG::ReportPort report;
  REQUIRE(report);
  CHECK(report.verbosity() == 1);
  CHECK_FALSE(report.log_file().has_value());
  CHECK_NOTHROW(report.info("automatic console fallback marker"));
  CHECK_NOTHROW(report.flush());
}

TEST_CASE("OutputSession owns a temporary default path", "[output][session][default]")
{
  std::filesystem::path root;
  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output;
    root = output.path().root();
    CHECK(std::filesystem::is_directory(root));
    const auto log_file = output.report_port().log_file();
    REQUIRE(log_file.has_value());
    CHECK(log_file->parent_path() == root);
  }
  CHECK_FALSE(std::filesystem::exists(root));
}

TEST_CASE("Configured OutputSession keeps data and logs in one path", "[output][session][layout]")
{
  auto parent = DiFfRG::OutputPath::temporary();
  const auto root = parent.root() / "configured";
  const auto json = output_json(root);
  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(json);
    CHECK(output.path().root() == std::filesystem::absolute(root).lexically_normal());
    CHECK(output.report_port().log_file() == std::optional(output.path().run_file(".log")));
    output.write_frame(0., [](auto &frame) { frame.table("values.csv").value("value", 1.); });
  }
  CHECK(std::filesystem::exists(root / "run.log"));
  CHECK(std::filesystem::exists(root / "run_values.csv"));
}

TEST_CASE("RunReporter aggregates progress every second below verbosity five", "[output][reporter][rate-limit]")
{
  auto path = DiFfRG::OutputPath::temporary();
  auto settings = output_settings();
  settings.verbosity = 2;

  DiFfRG::RunReporter reporter(path, settings, true);
  auto report = reporter.port();
  for (int i = 0; i < 3; ++i) {
    DiFfRG::ProgressEvent event;
    event.topic = DiFfRG::progress_topics::implicit_residual;
    event.time = 0.1 * i;
    event.duration_ms = 2. + i;
    event.field("rejects", i, 2).field("debug_only", i, 5);
    report.progress(event);
  }
  DiFfRG::ProgressEvent hidden;
  hidden.topic = DiFfRG::progress_topics::output;
  hidden.minimum_verbosity = 3;
  report.progress(hidden);
  DiFfRG::SummaryEvent summary{.component = "custom"};
  summary.timing("setup", 1.25, 2).timing("solve", 3.5, 4);
  report.summary(summary);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2500);
  std::string content;
  do {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    content = read_file(path.run_file(".log"));
  } while (content.find("[res ]") == std::string::npos && std::chrono::steady_clock::now() < deadline);

  REQUIRE_THAT(content, Catch::Matchers::ContainsSubstring("[res ]"));
  reporter.finish();

  content = read_file(path.run_file(".log"));
  CHECK(count_occurrences(content, "[res ]") == 1);
  CHECK_THAT(content, Catch::Matchers::ContainsSubstring("n=3"));
  CHECK_THAT(content, Catch::Matchers::ContainsSubstring("avg=3"));
  CHECK_THAT(content, Catch::Matchers::ContainsSubstring("wall="));
  CHECK_THAT(content, Catch::Matchers::ContainsSubstring("rejects="));
  CHECK(content.find("debug_only=") == std::string::npos);
  CHECK(content.find("[out ]") == std::string::npos);
  CHECK_THAT(content, Catch::Matchers::ContainsSubstring("[sum] custom setup=1.25000ms(2) solve=3.50000ms(4)"));
  for (std::istringstream lines(content); lines;) {
    std::string line;
    std::getline(lines, line);
    CHECK(line.size() <= 100);
  }
}

TEST_CASE("RunReporter leaves verbosity five unthrottled", "[output][reporter][verbosity]")
{
  auto path = DiFfRG::OutputPath::temporary();
  auto settings = output_settings();
  settings.verbosity = 5;

  {
    DiFfRG::RunReporter reporter(path, settings, true);
    for (int i = 0; i < 3; ++i) {
      DiFfRG::ProgressEvent event;
      event.topic = {"kry", "custom"};
      event.time = i;
      event.iterations = i + 1;
      reporter.port().progress(event);
    }
  }

  const auto content = read_file(path.run_file(".log"));
  CHECK(count_occurrences(content, "[kry ]") == 3);
  CHECK(count_occurrences(content, "custom") == 3);
  CHECK_THAT(content, Catch::Matchers::ContainsSubstring("unthrottled diagnostic output"));
}

TEST_CASE("RunReporter shutdown is safe during concurrent submissions", "[output][reporter][thread-safe]")
{
  auto path = DiFfRG::OutputPath::temporary();
  auto settings = output_settings();
  settings.verbosity = 2;

  DiFfRG::RunReporter reporter(path, settings, true);
  auto report = reporter.port();
  joining_thread producer([report] {
    for (int i = 0; i < 1000; ++i)
      report.progress(
          {.topic = DiFfRG::progress_topics::jacobian, .time = static_cast<double>(i), .minimum_verbosity = 2});
  });
  std::vector<joining_thread> finishers;
  for (int i = 0; i < 4; ++i)
    finishers.emplace_back([&reporter] { reporter.finish(); });

  producer.join();
  finishers.clear();
  CHECK_FALSE(report);
  CHECK_NOTHROW(report.info("ignored after shutdown"));
  CHECK_NOTHROW(reporter.finish());
}

TEST_CASE("OutputPath is a shared immutable ownership handle", "[output][path]")
{
  STATIC_CHECK_FALSE(std::is_default_constructible_v<DiFfRG::OutputPath>);
  STATIC_CHECK(std::is_copy_constructible_v<DiFfRG::OutputPath>);
  STATIC_CHECK(std::is_copy_assignable_v<DiFfRG::OutputPath>);
  STATIC_CHECK(std::is_move_constructible_v<DiFfRG::OutputPath>);
  STATIC_CHECK(std::is_move_assignable_v<DiFfRG::OutputPath>);
}

TEST_CASE("OutputPath copies share temporary cleanup ownership", "[output][path][temporary]")
{
  std::filesystem::path root;
  std::optional<DiFfRG::OutputPath> copy;
  {
    auto original = DiFfRG::OutputPath::temporary();
    root = original.root();
    copy = original;
  }
  CHECK(std::filesystem::is_directory(root));
  copy.reset();
  CHECK_FALSE(std::filesystem::exists(root));
}

TEST_CASE("OutputPath temporary cleanup and retention are explicit", "[output][path][temporary]")
{
  std::filesystem::path removed_path;
  {
    auto path = DiFfRG::OutputPath::temporary();
    removed_path = path.root();
    REQUIRE(std::filesystem::is_directory(removed_path));
  }
  CHECK_FALSE(std::filesystem::exists(removed_path));

  std::filesystem::path kept_path;
  {
    auto path = DiFfRG::OutputPath::temporary(DiFfRG::TemporaryRetention::keep);
    kept_path = path.root();
  }
  CHECK(std::filesystem::is_directory(kept_path));
  std::filesystem::remove_all(kept_path);
}

TEST_CASE("OutputPath moves transfer temporary cleanup ownership", "[output][path][temporary]")
{
  std::filesystem::path root;
  {
    auto original = DiFfRG::OutputPath::temporary();
    root = original.root();
    auto moved = std::move(original);
    CHECK(std::filesystem::is_directory(root));

    auto replacement = DiFfRG::OutputPath::temporary();
    replacement = std::move(moved);
    CHECK(std::filesystem::is_directory(root));
  }
  CHECK_FALSE(std::filesystem::exists(root));
}

TEST_CASE("OutputPath child shares its temporary root owner", "[output][path][temporary]")
{
  std::filesystem::path root;
  {
    auto child = [&]() {
      auto parent = DiFfRG::OutputPath::temporary();
      root = parent.root();
      return parent.child("tuning/run-1", "trial");
    }();
    CHECK(std::filesystem::is_directory(root));
    CHECK(child.root() == root / "tuning/run-1");
  }
  CHECK_FALSE(std::filesystem::exists(root));
}

TEST_CASE("The configuration copy is only written when nothing else records it", "[output][session][json]")
{
  using Session = DiFfRG::OutputSession_impl<0, dealii::Vector<double>>;

  const auto log_json = [](const DiFfRG::OutputPath &path) { return path.root() / (path.run_name() + ".log.json"); };

  SECTION("HDF5 off: the configuration has nowhere else to go, so it is written")
  {
    auto path = DiFfRG::OutputPath::temporary();
    auto settings = output_settings();
    REQUIRE_FALSE(settings.write_hdf5);
    {
      Session output(path, settings);
    }
    CHECK(std::filesystem::exists(log_json(path)));
  }

  SECTION("HDF5 on: the configuration travels in /config, so no separate copy")
  {
    auto path = DiFfRG::OutputPath::temporary();
    auto settings = output_settings();
    settings.write_hdf5 = true;
    {
      Session output(path, settings);
    }
    CHECK_FALSE(std::filesystem::exists(log_json(path)));
  }

  SECTION("/output/json forces the copy even with HDF5 on")
  {
    auto path = DiFfRG::OutputPath::temporary();
    auto settings = output_settings();
    settings.write_hdf5 = true;
    settings.write_json = true;
    {
      Session output(path, settings);
    }
    CHECK(std::filesystem::exists(log_json(path)));
  }

  SECTION("/output/json is read from the configuration and defaults to off")
  {
    auto parent = DiFfRG::OutputPath::temporary();
    auto json = output_json(parent.root());
    CHECK_FALSE(DiFfRG::Config::OutputSettings(json).write_json);

    json().as_object().at("output").as_object()["json"] = true;
    CHECK(DiFfRG::Config::OutputSettings(json).write_json);
  }
}

TEST_CASE("The session records how the run ended", "[output][session][hdf5][status]")
{
  using Session = DiFfRG::OutputSession_impl<0, dealii::Vector<double>>;

  const auto status = [](const DiFfRG::OutputPath &path) {
    auto file =
        DiFfRG::hdf5::File::open((path.root() / (path.run_name() + ".h5")).string(), DiFfRG::hdf5::Access::ReadOnly);
    auto root = file.root();
    return std::pair{root.read_attribute<int>("finished"), root.read_attribute<int>("crashed")};
  };

  const auto hdf5_settings = [] {
    auto settings = output_settings();
    settings.write_hdf5 = true;
    return settings;
  };

  SECTION("an explicit finish() is a clean end")
  {
    auto path = DiFfRG::OutputPath::temporary();
    {
      Session output(path, hdf5_settings());
      output.finish();
    }
    CHECK(status(path) == std::pair{1, 0});
  }

  SECTION("falling off the end of the scope is also a clean end")
  {
    auto path = DiFfRG::OutputPath::temporary();
    {
      Session output(path, hdf5_settings());
    }
    CHECK(status(path) == std::pair{1, 0});
  }

  SECTION("unwinding through the destructor is finished, but crashed")
  {
    auto path = DiFfRG::OutputPath::temporary();
    try {
      Session output(path, hdf5_settings());
      throw std::runtime_error("the timestepper gave up");
    } catch (const std::runtime_error &) {
    }
    // The session still flushed and closed its files -- that is the point of the distinction.
    CHECK(status(path) == std::pair{1, 1});
  }

  SECTION("an explicit finish() before the throw still counts as clean")
  {
    auto path = DiFfRG::OutputPath::temporary();
    try {
      Session output(path, hdf5_settings());
      output.finish();
      throw std::runtime_error("thrown after the outputs were closed");
    } catch (const std::runtime_error &) {
    }
    CHECK(status(path) == std::pair{1, 0});
  }
}

TEST_CASE("OutputSettings keeps debugging and memory policy in C++", "[output][settings][json]")
{
  auto parent = DiFfRG::OutputPath::temporary();
  auto json = output_json(parent.root());
  auto &output = json().as_object().at("output").as_object();
  output["async"] = false;
  output["max_pending_bytes"] = 1;

  const DiFfRG::Config::OutputSettings settings(json);
  CHECK(settings.asynchronous);
  CHECK(settings.max_pending_bytes == DiFfRG::Config::OutputSettings::default_max_pending_bytes);
}

TEST_CASE("OutputPath creates unique temporary roots concurrently", "[output][path][thread-safe]")
{
  std::mutex mutex;
  std::vector<std::filesystem::path> roots;
  std::vector<joining_thread> threads;
  for (unsigned int i = 0; i < 16; ++i)
    threads.emplace_back([&]() {
      auto path = DiFfRG::OutputPath::temporary();
      std::scoped_lock lock(mutex);
      roots.push_back(path.root());
    });
  threads.clear();
  CHECK(std::set(roots.begin(), roots.end()).size() == roots.size());
}

TEST_CASE("OutputPath JSON constructor forwards to the persistent layout", "[output][path][json]")
{
  auto temporary_parent = DiFfRG::OutputPath::temporary();
  const auto configured_root = temporary_parent.root() / "configured";
  auto json = output_json(configured_root);
  {
    DiFfRG::OutputPath typed(configured_root, "run", "run");
    DiFfRG::OutputPath path(json);
    CHECK(path.root() == typed.root());
    CHECK(path.run_name() == typed.run_name());
    CHECK(path.field_directory() == typed.field_directory());
    CHECK(path.root() == std::filesystem::absolute(configured_root).lexically_normal());
    CHECK(path.run_name() == "run");
    CHECK(path.field_directory() == "run");
    std::filesystem::create_directories(path.root());
  }
  CHECK(std::filesystem::is_directory(configured_root));
}

TEST_CASE("OutputPath rejects unsafe paths and failed artifact copies", "[output][path][safety]")
{
  CHECK_THROWS_AS(DiFfRG::OutputPath({}, "run"), std::invalid_argument);
  CHECK_THROWS_AS(DiFfRG::OutputPath("output", ""), std::invalid_argument);
  CHECK_THROWS_AS(DiFfRG::OutputPath("output", "nested/run"), std::invalid_argument);
  CHECK_THROWS_AS(DiFfRG::OutputPath("output", "../run"), std::invalid_argument);
  CHECK_THROWS_AS(DiFfRG::OutputPath("output", "run", "../fields"), std::invalid_argument);

  auto path = DiFfRG::OutputPath::temporary();
  CHECK_THROWS_AS(path.resolve("../escape"), std::invalid_argument);
  CHECK_THROWS_AS(path.resolve(std::filesystem::current_path()), std::invalid_argument);
  CHECK_THROWS_AS(path.run_file("csv"), std::invalid_argument);
  CHECK_THROWS_AS(path.run_file(".dir/file"), std::invalid_argument);
  CHECK_THROWS_AS(path.child("../escape", "child"), std::invalid_argument);
  CHECK_THROWS_WITH(path.copy_tree_from(path.root() / "missing"), Catch::Matchers::ContainsSubstring("does not exist"));
}
