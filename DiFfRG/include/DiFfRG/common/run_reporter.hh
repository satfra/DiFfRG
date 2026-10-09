#pragma once

#include <DiFfRG/discretization/data/output_path.hh>
#include <DiFfRG/discretization/data/output_settings.hh>

#include <spdlog/common.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace DiFfRG
{
  /**
   * A stable progress category used for aggregation and rendering. Both views must
   * refer to strings with static storage duration.
   */
  struct ProgressTopic {
    std::string_view tag;
    std::string_view stage;
    auto operator<=>(const ProgressTopic &) const = default;
  };

  namespace progress_topics
  {
    inline constexpr ProgressTopic timestep{"step", {}};
    inline constexpr ProgressTopic explicit_residual{"res", "explicit"};
    inline constexpr ProgressTopic implicit_residual{"res", "implicit"};
    inline constexpr ProgressTopic variables{"vars", {}};
    inline constexpr ProgressTopic jacobian{"jac", {}};
    inline constexpr ProgressTopic trapezoidal_jacobian{"jac", "TR"};
    inline constexpr ProgressTopic bdf2_jacobian{"jac", "BDF2"};
    inline constexpr ProgressTopic factorization{"fac", {}};
    inline constexpr ProgressTopic implicit_linear_solve{"lin", "implicit"};
    inline constexpr ProgressTopic output{"out", {}};
  } // namespace progress_topics

  struct ProgressField {
    std::string_view name;
    double value = 0.;
    int minimum_verbosity = 1;
  };

  /** A small, allocation-free runtime progress record. Field names must have static storage duration. */
  struct ProgressEvent {
    ProgressTopic topic = progress_topics::timestep;
    double time = 0.;
    double duration_ms = -1.;
    long int iterations = -1;
    int minimum_verbosity = 1;
    std::array<ProgressField, 16> fields{};
    std::size_t field_count = 0;

    ProgressEvent &field(const std::string_view name, const double value, const int minimum_verbosity = 1)
    {
      if (field_count < fields.size()) fields[field_count++] = {name, value, minimum_verbosity};
      return *this;
    }
  };

  struct SummaryMetric {
    std::string_view name;
    double average_ms = 0.;
    std::size_t calls = 0;
  };

  /** A compact component summary. Metric names and the component name must have static storage duration. */
  struct SummaryEvent {
    std::string_view component;
    std::array<SummaryMetric, 4> metrics{};
    std::size_t metric_count = 0;

    SummaryEvent &timing(const std::string_view name, const double average_ms, const std::size_t calls)
    {
      if (metric_count < metrics.size()) metrics[metric_count++] = {name, average_ms, calls};
      return *this;
    }
  };

  namespace internal
  {
    class ReporterState;
    class ReporterSlot;
  } // namespace internal

  /**
   * Copyable, thread-safe handle to shared reporter state. A default port lazily creates a console-only reporter;
   * ports returned by RunReporter share its configured console and optional file destinations.
   */
  class ReportPort
  {
  public:
    /** Create a lazily initialized console-only reporter. */
    ReportPort();
    ReportPort(const ReportPort &) = default;
    ReportPort &operator=(const ReportPort &) = default;

    template <typename... Args> void info(spdlog::format_string_t<Args...> format, Args &&...args) const
    {
      info(spdlog::fmt_lib::format(format, std::forward<Args>(args)...));
    }
    template <typename... Args> void warn(spdlog::format_string_t<Args...> format, Args &&...args) const
    {
      warn(spdlog::fmt_lib::format(format, std::forward<Args>(args)...));
    }
    template <typename... Args> void error(spdlog::format_string_t<Args...> format, Args &&...args) const
    {
      error(spdlog::fmt_lib::format(format, std::forward<Args>(args)...));
    }
    template <typename... Args> void debug(spdlog::format_string_t<Args...> format, Args &&...args) const
    {
      debug(spdlog::fmt_lib::format(format, std::forward<Args>(args)...));
    }

    void info(const std::string &message) const;
    void warn(const std::string &message) const;
    void error(const std::string &message) const;
    void debug(const std::string &message) const;
    void progress(const ProgressEvent &event) const;
    void summary(const SummaryEvent &event) const;
    void flush() const;
    /** Path of the duplicated text log, or no value for a console-only reporter. */
    std::optional<std::filesystem::path> log_file() const;

    int verbosity() const noexcept;
    double Lambda() const noexcept;
    explicit operator bool() const noexcept;

  private:
    explicit ReportPort(std::shared_ptr<internal::ReporterState> state);
    std::shared_ptr<internal::ReporterSlot> slot;
    friend class RunReporter;
  };

  struct RunReporterOptions {
    /** Tags this reporter's lines in the shared run log, e.g. `[quadrature]`. */
    std::string reporter_name = "run";
    /** Whether to echo to the console. Reporters that would interleave with the run progress
     * display (e.g. the quadrature log) set this to false and write only to the run log. */
    bool console = true;
  };

  /**
   * Owns the ordered severity sinks and coalesced progress state for one simulation run.
   *
   * Every file-backed reporter writes into the one `<run name>.log`; reporters on the same path
   * share a single process-wide file sink, so a reporter opened before the session (the
   * QuadratureProvider's, typically) is appended to rather than truncated away.
   */
  class RunReporter
  {
  public:
    RunReporter() = default;
    RunReporter(OutputPath path, const Config::OutputSettings &settings, bool active, RunReporterOptions options = {});
    ~RunReporter() noexcept;

    RunReporter(const RunReporter &) = delete;
    RunReporter &operator=(const RunReporter &) = delete;
    RunReporter(RunReporter &&) noexcept = default;
    RunReporter &operator=(RunReporter &&) noexcept = default;

    ReportPort port() const;
    void finish();

  private:
    std::shared_ptr<internal::ReporterState> state;
  };
} // namespace DiFfRG
