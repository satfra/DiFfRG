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

/**
 * @file run_reporter.hh
 * @brief Log messages and progress lines of a run.
 *
 * Code that wants to log something holds a ReportPort, usually obtained from the timestepper, assembler or output
 * session, and calls `info`, `warn`, `error` or `debug` with an fmt-style format string:
 * @code
 * log.info("converged after {} iterations", n);
 * @endcode
 *
 * Where messages go is set by two parameters:
 * - `/output/verbosity` (int, default 0) for the console: 0 shows warnings and errors, 1 also info, 2 and more
 *   also debug messages. Progress lines appear from 1; their detail grows with the verbosity. From 5 on every
 *   progress event is printed immediately (slow), below that they are collected and printed about once per
 *   second.
 * - `/output/log_level` (int, default 2 = info) for the log file `<output folder>/<run name>.log`, using spdlog's
 *   numbering: 0 trace, 1 debug, 2 info, 3 warn, 4 error, 5 critical, 6 off.
 */

namespace DiFfRG
{
  /**
   * @brief A progress category, e.g. "residual of the implicit stepper". Used to group progress lines.
   *
   * Both views must refer to strings with static storage duration.
   */
  struct ProgressTopic {
    std::string_view tag;
    std::string_view stage;
    auto operator<=>(const ProgressTopic &) const = default;
  };

  /// The progress categories used by the timesteppers.
  namespace progress_topics
  {
    inline constexpr ProgressTopic timestep{"step", {}};
    /// One explicit multistep step: its size and local error estimate.
    inline constexpr ProgressTopic explicit_step{"step", "explicit"};
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

  /// A named number attached to a ProgressEvent. The name must have static storage duration.
  struct ProgressField {
    std::string_view name;
    double value = 0.;
    int minimum_verbosity = 1;
  };

  /** @brief One progress record, e.g. one residual evaluation. Field names must have static storage duration. */
  struct ProgressEvent {
    /// The category.
    ProgressTopic topic = progress_topics::timestep;
    /// RG time \f$t\f$; the line also shows \f$k = \Lambda e^{-t}\f$ if `/physical/Lambda` is set.
    double time = 0.;
    /// Duration in milliseconds, or -1 if not given.
    double duration_ms = -1.;
    /// Iteration count, or -1 if not given.
    long int iterations = -1;
    /// Only shown when `/output/verbosity` is at least this.
    int minimum_verbosity = 1;
    /// Additional named values.
    std::array<ProgressField, 16> fields{};
    /// Number of used entries in `fields`.
    std::size_t field_count = 0;

    /// Attach a named value; fields beyond the 16th are dropped.
    ProgressEvent &field(const std::string_view name, const double value, const int minimum_verbosity = 1)
    {
      if (field_count < fields.size()) fields[field_count++] = {name, value, minimum_verbosity};
      return *this;
    }
  };

  /// One timing in a SummaryEvent. The name must have static storage duration.
  struct SummaryMetric {
    std::string_view name;
    double average_ms = 0.;
    std::size_t calls = 0;
  };

  /** @brief Timing summary of one component. Metric names and the component name must have static storage duration. */
  struct SummaryEvent {
    /// Name of the component, e.g. "assembler".
    std::string_view component;
    /// The timings.
    std::array<SummaryMetric, 4> metrics{};
    /// Number of used entries in `metrics`.
    std::size_t metric_count = 0;

    /// Add a timing; timings beyond the 4th are dropped.
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
   * @brief Where a component sends its log messages and progress. Cheap to copy and thread-safe.
   *
   * A default-constructed port prints to the console only, with verbosity 1, and sets itself up on first use. Ports
   * obtained from a RunReporter write to that run's console and log file. See run_reporter.hh for the parameters
   * that control the output.
   */
  class ReportPort
  {
  public:
    /** A console-only port, set up on first use. */
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

    /// Log an ordinary message.
    void info(const std::string &message) const;
    /// Log a warning.
    void warn(const std::string &message) const;
    /// Log an error.
    void error(const std::string &message) const;
    /// Log a debug message.
    void debug(const std::string &message) const;
    /// Report progress; see ProgressEvent.
    void progress(const ProgressEvent &event) const;
    /// Report a timing summary; see SummaryEvent.
    void summary(const SummaryEvent &event) const;
    /// Write out everything queued so far.
    void flush() const;
    /** Path of the log file, or no value for a console-only port. */
    std::optional<std::filesystem::path> log_file() const;

    /// The console verbosity, `/output/verbosity`.
    int verbosity() const noexcept;
    /// `/physical/Lambda`, used to print \f$k\f$ in progress lines; -1 if unset.
    double Lambda() const noexcept;
    /// Whether messages sent to this port are written anywhere.
    explicit operator bool() const noexcept;

  private:
    explicit ReportPort(std::shared_ptr<internal::ReporterState> state);
    std::shared_ptr<internal::ReporterSlot> slot;
    friend class RunReporter;
  };

  /// Options of a RunReporter.
  struct RunReporterOptions {
    /** Tags this reporter's lines in the shared run log, e.g. `[quadrature]`. */
    std::string reporter_name = "run";
    /** Whether to echo to the console. Reporters that would interleave with the run progress
     * display (e.g. the quadrature log) set this to false and write only to the run log. */
    bool console = true;
  };

  /**
   * @brief The console and log-file output of one simulation run; hands out ReportPort%s.
   *
   * Every reporter of a run writes into the one `<run name>.log`; reporters on the same path share one file, which is
   * emptied when it is first opened in a process and appended to afterwards.
   */
  class RunReporter
  {
  public:
    /// A reporter that discards everything; its port() is inactive.
    RunReporter() = default;
    /**
     * @brief Set up the output of a run.
     *
     * @param path Where the log file goes: `<root>/<run name>.log`.
     * @param settings Verbosity, log level and \f$\Lambda\f$, read from `/output/...` and `/physical/Lambda`.
     * @param active If false, all messages are discarded and no log file is created.
     * @param options Name tag and console switch.
     */
    RunReporter(OutputPath path, const Config::OutputSettings &settings, bool active, RunReporterOptions options = {});
    /// Calls finish().
    ~RunReporter() noexcept;

    RunReporter(const RunReporter &) = delete;
    RunReporter &operator=(const RunReporter &) = delete;
    RunReporter(RunReporter &&) noexcept = default;
    RunReporter &operator=(RunReporter &&) noexcept = default;

    /// A port writing to this run's output.
    ReportPort port() const;
    /// Write out all pending output and stop accepting messages. Safe to call more than once.
    void finish();

  private:
    std::shared_ptr<internal::ReporterState> state;
  };
} // namespace DiFfRG
