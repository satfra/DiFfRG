#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/discretization/data/output_path.hh>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

using namespace DiFfRG;

namespace
{
  DiFfRG::ConfigTree quadrature_log_json(const std::filesystem::path &folder)
  {
    return DiFfRG::json::value({{"output", {{"folder", folder.string()}, {"name", "run"}}}});
  }

  std::string read_file(const std::filesystem::path &path)
  {
    std::ifstream stream(path);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
  }
} // namespace

TEST_CASE("Test quadrature provider", "[double][quadrature]")
{
  DiFfRG::Init();

  const double T = GENERATE(take(1, random(0.1, 1.0)));
  const double start_k = GENERATE(take(1, random(1.0, 100.0)));
  const double speed_k = GENERATE(take(2, random(1e-5, 1e-2)));
  const int steps = GENERATE(take(2, random(1e+1, 1e+4)));

  const size_t order = GENERATE(take(2, random(8, 64)));

  DiFfRG::QuadratureProvider quadrature_provider;

  SECTION("Bare test")
  {
    double point = 0;
    double weight = 0;

    for (int j = 0; j < steps; ++j) {
      const double k = std::exp((long double)(-j * speed_k)) * start_k;

      const auto nodes = quadrature_provider.template nodes<double, CPU_memory>(order);
      const auto matsubara_nodes = quadrature_provider.template matsubara_nodes<double, CPU_memory>(T, k);
      const auto weights = quadrature_provider.template weights<double, CPU_memory>(order);
      const auto matsubara_weights = quadrature_provider.template matsubara_weights<double, CPU_memory>(T, k);

      for (uint i = 0; i < order; ++i) {
        point += nodes[i] / (double)steps;
        weight += weights[i] / (double)steps;
        if (!isfinite(nodes[i]) || !isfinite(weights[i]))
          throw std::runtime_error("Not finite: Quadrature nodes or weights");
      }
      for (uint i = 0; i < matsubara_nodes.size(); ++i) {
        point += matsubara_nodes[i] / (double)steps;
        weight += matsubara_weights[i] / (double)steps;

        if (!isfinite(matsubara_nodes[i]) || !isfinite(matsubara_weights[i])) {
          std::string error =
              "Not finite: Matsubara nodes or weights for k = " + std::to_string(k) + " and T = " + std::to_string(T);
          error += "\nMatsubara size is " + std::to_string(matsubara_nodes.size());
          throw std::runtime_error(error);
        }
      }
    }

    REQUIRE(point != Catch::Approx(0.));
    REQUIRE(weight != Catch::Approx(0.));
  }
}

TEST_CASE("Quadrature provider without an output folder stays silent", "[quadrature][output][default]")
{
  DiFfRG::Init();

  const DiFfRG::ConfigTree config(DiFfRG::json::value({{"output", {{"name", "run"}}}}));
  DiFfRG::QuadratureProvider quadrature_provider(config);
  const auto nodes = quadrature_provider.template nodes<double, CPU_memory>(32);
  REQUIRE(nodes.size() == 32);
}

TEST_CASE("Quadrature provider reports into the run log", "[quadrature][output]")
{
  DiFfRG::Init();

  auto path = DiFfRG::OutputPath::temporary(TemporaryRetention::remove_on_destruction, "run", "run");
  const auto run_log = path.run_file(".log");

  SECTION("Enabled by the configured output folder")
  {
    {
      DiFfRG::QuadratureProvider quadrature_provider(quadrature_log_json(path.root()));
      // Requesting a quadrature has to be reported, since this is what the records exist for. The provider owns the
      // reporter, so they are only guaranteed on disk once it has been destroyed.
      const auto nodes = quadrature_provider.template nodes<double, CPU_memory>(32);
      REQUIRE(nodes.size() == 32);
    }

    REQUIRE(std::filesystem::exists(run_log));
    const auto contents = read_file(run_log);
    CHECK_THAT(contents, Catch::Matchers::ContainsSubstring("Initialized quadrature provider"));
    CHECK_THAT(contents, Catch::Matchers::ContainsSubstring("order = 32"));
    // The records carry their own reporter name, so they stay distinguishable from the session's.
    CHECK_THAT(contents, Catch::Matchers::ContainsSubstring("[quadrature]"));
    // No side-channel companion is left behind.
    CHECK_FALSE(std::filesystem::exists(path.run_file("_quadrature", ".log")));
  }

  SECTION("An explicit port takes precedence over the own reporter")
  {
    {
      Config::OutputSettings settings;
      RunReporter run_reporter(path, settings, true);

      DiFfRG::QuadratureProvider quadrature_provider(quadrature_log_json(path.root()), run_reporter.port());
      const auto nodes = quadrature_provider.template nodes<double, CPU_memory>(32);
      REQUIRE(nodes.size() == 32);
    }

    const auto contents = read_file(run_log);
    CHECK_THAT(contents, Catch::Matchers::ContainsSubstring("order = 32"));
    // The messages belong to the port that was handed in, so the provider never opens a reporter of its own.
    CHECK_THAT(contents, !Catch::Matchers::ContainsSubstring("[quadrature]"));
  }

  SECTION("A session opened afterwards does not truncate the quadrature records")
  {
    {
      DiFfRG::QuadratureProvider quadrature_provider(quadrature_log_json(path.root()));
      const auto nodes = quadrature_provider.template nodes<double, CPU_memory>(32);
      REQUIRE(nodes.size() == 32);

      // This is the ordering of every example main: the provider is built with the flows, the session only later.
      Config::OutputSettings settings;
      RunReporter session_reporter(path, settings, true);
      session_reporter.port().info("session marker");
    }

    const auto contents = read_file(run_log);
    CHECK_THAT(contents, Catch::Matchers::ContainsSubstring("order = 32"));
    CHECK_THAT(contents, Catch::Matchers::ContainsSubstring("session marker"));
  }
}
