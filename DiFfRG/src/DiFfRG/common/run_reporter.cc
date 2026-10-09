#include <DiFfRG/common/run_reporter.hh>

#include <DiFfRG/common/utils.hh>

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iomanip>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <vector>

namespace DiFfRG::internal
{
  namespace
  {
    constexpr auto console_interval = std::chrono::seconds(1);
    constexpr int unthrottled_verbosity = 5;

    /**
     * One file sink per path, shared process-wide. A QuadratureProvider opens the run log from
     * within the integrator constructors, long before the OutputSession that later writes into
     * it; two independent sinks on one path would each keep their own file offset and overwrite
     * each other. The shared _mt sink serializes their records behind its own mutex.
     *
     * A path is truncated the first time it is opened in a process and appended to afterwards, so
     * a second reporter (or session) writing the same file does not discard what the first one
     * recorded.
     */
    spdlog::sink_ptr shared_file_sink(const std::filesystem::path &file)
    {
      static std::mutex mutex;
      static std::map<std::string, std::weak_ptr<spdlog::sinks::basic_file_sink_mt>> sinks;

      std::error_code ec;
      auto key = std::filesystem::weakly_canonical(file, ec).string();
      if (ec) key = file.string();

      const std::lock_guard<std::mutex> lock(mutex);
      const auto it = sinks.find(key);
      if (it != sinks.end())
        if (auto live = it->second.lock()) return live;

      // A known key that no longer resolves was opened earlier in this process and must not be
      // truncated again.
      const bool truncate = it == sinks.end();
      auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file.string(), truncate);
      sinks[key] = sink;
      return sink;
    }

    struct Aggregate {
      ProgressEvent latest;
      std::size_t calls = 0;
      std::size_t duration_samples = 0;
      double duration_sum_ms = 0.;
      long int iteration_sum = 0;
      std::size_t iteration_samples = 0;

      void add(const ProgressEvent &event)
      {
        latest = event;
        ++calls;
        if (event.duration_ms >= 0.) {
          duration_sum_ms += event.duration_ms;
          ++duration_samples;
        }
        if (event.iterations >= 0) {
          iteration_sum += event.iterations;
          ++iteration_samples;
        }
      }
    };

    std::string compact_number(const double value)
    {
      std::ostringstream out;
      const double magnitude = std::abs(value);
      if ((magnitude > 0. && magnitude < 1e-3) || magnitude >= 1e5)
        out << std::scientific << std::setprecision(3) << value;
      else
        out << std::fixed << std::setprecision(5) << value;
      return out.str();
    }

    std::vector<std::string> render(const Aggregate &aggregate, const double Lambda, const int verbosity,
                                    const bool individual, const std::size_t wall_ms)
    {
      const auto &event = aggregate.latest;
      std::ostringstream line;
      line << '[' << std::setw(4) << std::left << event.topic.tag << "] t=" << compact_number(event.time);
      if (Lambda > 0.) line << " k=" << compact_number(std::exp(-event.time) * Lambda);
      // Total wall time of the run, not the cost of the reported operation (that is avg=).
      line << " wall=" << time_format_ms(wall_ms);
      if (!event.topic.stage.empty()) line << " " << event.topic.stage;
      if (!individual && aggregate.calls > 1) line << " n=" << aggregate.calls;
      if (aggregate.duration_samples > 0) {
        const double mean = aggregate.duration_sum_ms / static_cast<double>(aggregate.duration_samples);
        line << " avg=" << std::setprecision(4) << mean << "ms";
      }
      if (aggregate.iteration_samples > 0) {
        const double mean = static_cast<double>(aggregate.iteration_sum) / aggregate.iteration_samples;
        line << " it=" << std::setprecision(3) << mean;
      }

      std::vector<std::string> lines{line.str()};
      bool has_details = false;
      for (std::size_t i = 0; i < event.field_count; ++i)
        has_details = has_details || event.fields[i].minimum_verbosity <= verbosity;
      if (has_details) {
        std::ostringstream details;
        details << "       ";
        bool first = true;
        for (std::size_t i = 0; i < event.field_count; ++i) {
          if (event.fields[i].minimum_verbosity > verbosity) continue;
          if (!first) details << " | ";
          details << event.fields[i].name << '=' << compact_number(event.fields[i].value);
          first = false;
        }
        auto text = details.str();
        if (text.size() > 100) text.resize(100);
        lines.push_back(std::move(text));
      }
      if (lines.front().size() > 100) lines.front().resize(100);
      return lines;
    }

    std::vector<std::string> render(const SummaryEvent &event)
    {
      std::vector<std::string> lines{"[sum] " + std::string(event.component)};
      for (std::size_t i = 0; i < event.metric_count; ++i) {
        const auto &metric = event.metrics[i];
        const std::string text = std::string(metric.name) + '=' + compact_number(metric.average_ms) + "ms(" +
                                 std::to_string(metric.calls) + ')';
        if (lines.back().size() + text.size() + 1 <= 100)
          lines.back() += ' ' + text;
        else if (lines.size() == 1)
          lines.push_back("      " + text);
      }
      if (lines.back().size() > 100) lines.back().resize(100);
      return lines;
    }

    struct Destination {
      std::shared_ptr<spdlog::logger> logger;
      std::map<ProgressTopic, Aggregate> pending;
    };

    struct PendingBatch {
      std::shared_ptr<spdlog::logger> logger;
      std::map<ProgressTopic, Aggregate> events;
    };
  } // namespace

  class ReporterState
  {
  public:
    ReporterState(const Config::OutputSettings &settings, const bool active, RunReporterOptions options)
        : verbosity_(settings.verbosity), Lambda_(settings.Lambda), accepting(active)
    {
      initialize(nullptr, settings, std::move(options));
    }

    ReporterState(OutputPath &&path, const Config::OutputSettings &settings, const bool active,
                  RunReporterOptions options)
        : owned_path(std::move(path)), verbosity_(settings.verbosity), Lambda_(settings.Lambda), accepting(active)
    {
      initialize(&*owned_path, settings, std::move(options));
    }

    ~ReporterState() noexcept
    {
      try {
        finish();
      } catch (...) {
      }
    }

    void message(const spdlog::level::level_enum level, const std::string &text)
    {
      if (!accepting.load()) return;
      for (const auto &destination : destinations)
        destination.logger->log(level, text);
    }

    void progress(const ProgressEvent &event)
    {
      if (!accepting.load() || verbosity_ < event.minimum_verbosity) return;
      (this->*progress_strategy)(event);
    }

    void summary(const SummaryEvent &event)
    {
      if (!accepting.load() || event.component.empty()) return;
      for (const auto &line : render(event))
        for (const auto &destination : destinations)
          destination.logger->info(line);
    }

    void flush()
    {
      for (const auto &destination : destinations)
        destination.logger->flush();
    }

    void finish()
    {
      {
        std::unique_lock lock(mutex);
        if (finished) return;
        if (stopped) {
          condition.wait(lock, [this] { return finished; });
          return;
        }
        accepting.store(false);
        stopped = true;
      }
      condition.notify_all();
      if (worker.joinable()) worker.join();
      flush_pending(true);
      flush();
      {
        std::scoped_lock lock(mutex);
        finished = true;
      }
      condition.notify_all();
    }

    int verbosity() const noexcept { return verbosity_; }
    double Lambda() const noexcept { return Lambda_; }
    bool active() const noexcept { return accepting.load(); }
    const std::optional<std::filesystem::path> &log_file() const noexcept { return log_file_; }

  private:
    void initialize(const OutputPath *path, const Config::OutputSettings &settings, RunReporterOptions options)
    {
      if (!accepting.load()) return;
      if (path != nullptr) create_folder(path->root().string());
      const auto queue_size = std::max<std::size_t>(256, settings.log_queue_size);
      thread_pool = std::make_shared<spdlog::details::thread_pool>(queue_size, 1);

      if (options.console) {
        auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sink->set_pattern("%v");
        auto logger = std::make_shared<spdlog::async_logger>(options.reporter_name + ".console", sink, thread_pool,
                                                             spdlog::async_overflow_policy::block);
        // The console follows /output/verbosity; only the file destination keeps the full log_level record.
        const auto console_level = verbosity_ <= 0   ? spdlog::level::warn
                                   : verbosity_ == 1 ? spdlog::level::info
                                                     : spdlog::level::debug;
        logger->set_level(console_level);
        destinations.push_back({std::move(logger), {}});
      }
      if (path != nullptr) {
        log_file_ = path->run_file(".log");
        // Named after the reporter, not "<name>.file": the shared sink's default pattern stamps
        // the logger name on every line, which is what keeps e.g. [quadrature] records
        // distinguishable from the session's inside the one run log.
        auto file_logger = std::make_shared<spdlog::async_logger>(options.reporter_name, shared_file_sink(*log_file_),
                                                                  thread_pool, spdlog::async_overflow_policy::block);
        file_logger->set_level(settings.log_level);
        file_logger->flush_on(spdlog::level::warn);
        destinations.push_back({std::move(file_logger), {}});
      }

      progress_strategy =
          verbosity_ >= unthrottled_verbosity ? &ReporterState::submit_immediate : &ReporterState::submit_aggregated;
      worker = std::thread([this] { run(); });
      if (verbosity_ >= unthrottled_verbosity)
        message(spdlog::level::warn,
                "[warn] verbosity=5 enables unthrottled diagnostic output and may significantly reduce performance");
    }
    void run()
    {
      std::unique_lock lock(mutex);
      while (!condition.wait_for(lock, console_interval, [this] { return stopped; })) {
        lock.unlock();
        flush_pending(false);
        lock.lock();
      }
    }

    void flush_pending(const bool final)
    {
      std::vector<PendingBatch> batches;
      {
        std::scoped_lock lock(mutex);
        for (auto &destination : destinations) {
          batches.push_back({destination.logger, {}});
          batches.back().events.swap(destination.pending);
        }
      }
      for (const auto &batch : batches) {
        for (const auto &[topic, aggregate] : batch.events)
          emit(batch.logger, aggregate, false);
        batch.logger->flush();
      }
    }

    void submit_immediate(const ProgressEvent &event)
    {
      Aggregate aggregate;
      aggregate.add(event);
      for (const auto &destination : destinations)
        emit(destination.logger, aggregate, true);
    }

    void submit_aggregated(const ProgressEvent &event)
    {
      std::scoped_lock lock(mutex);
      if (!accepting.load()) return;
      for (auto &destination : destinations)
        destination.pending[event.topic].add(event);
    }

    void emit(const std::shared_ptr<spdlog::logger> &logger, const Aggregate &aggregate, const bool individual) const
    {
      for (const auto &line : render(aggregate, Lambda_, verbosity_, individual, elapsed_ms()))
        logger->info(line);
    }

    std::size_t elapsed_ms() const
    {
      return static_cast<std::size_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count());
    }

    using ProgressStrategy = void (ReporterState::*)(const ProgressEvent &);

    // A file-backed reporter owns its path. This member is declared first so it
    // is destroyed after every sink has been flushed and released.
    std::optional<OutputPath> owned_path;
    std::optional<std::filesystem::path> log_file_;
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    int verbosity_ = 0;
    double Lambda_ = -1.;
    std::atomic_bool accepting = false;
    bool stopped = false;
    bool finished = false;
    ProgressStrategy progress_strategy = &ReporterState::submit_aggregated;
    std::shared_ptr<spdlog::details::thread_pool> thread_pool;
    std::vector<Destination> destinations;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable condition;
  };

  class ReporterSlot
  {
  public:
    ReporterSlot() = default;
    explicit ReporterSlot(std::shared_ptr<ReporterState> state) : state_(std::move(state)), fallback(false) {}

    std::shared_ptr<ReporterState> state() const
    {
      std::scoped_lock lock(mutex);
      if (!state_ && fallback) {
        Config::OutputSettings settings;
        settings.verbosity = 1;
        state_ = std::make_shared<ReporterState>(settings, true, RunReporterOptions{.reporter_name = "fallback"});
      }
      return state_;
    }

    int verbosity() const noexcept
    {
      const auto current = state_without_creation();
      return current ? current->verbosity() : (fallback ? 1 : 0);
    }

    double Lambda() const noexcept
    {
      const auto current = state_without_creation();
      return current ? current->Lambda() : -1.;
    }

    bool active() const noexcept
    {
      const auto current = state_without_creation();
      return current ? current->active() : fallback;
    }

    std::optional<std::filesystem::path> log_file() const noexcept
    {
      const auto current = state_without_creation();
      return current ? current->log_file() : std::nullopt;
    }

  private:
    std::shared_ptr<ReporterState> state_without_creation() const noexcept
    {
      std::scoped_lock lock(mutex);
      return state_;
    }

    mutable std::mutex mutex;
    mutable std::shared_ptr<ReporterState> state_;
    bool fallback = true;
  };
} // namespace DiFfRG::internal

namespace DiFfRG
{
  ReportPort::ReportPort() : slot(std::make_shared<internal::ReporterSlot>()) {}

  ReportPort::ReportPort(std::shared_ptr<internal::ReporterState> state)
      : slot(std::make_shared<internal::ReporterSlot>(std::move(state)))
  {
  }

  void ReportPort::info(const std::string &message) const
  {
    if (const auto state = slot->state()) state->message(spdlog::level::info, message);
  }
  void ReportPort::warn(const std::string &message) const
  {
    if (const auto state = slot->state()) state->message(spdlog::level::warn, message);
  }
  void ReportPort::error(const std::string &message) const
  {
    if (const auto state = slot->state()) state->message(spdlog::level::err, message);
  }
  void ReportPort::debug(const std::string &message) const
  {
    if (const auto state = slot->state()) state->message(spdlog::level::debug, message);
  }
  void ReportPort::progress(const ProgressEvent &event) const
  {
    if (const auto state = slot->state()) state->progress(event);
  }
  void ReportPort::summary(const SummaryEvent &event) const
  {
    if (const auto state = slot->state()) state->summary(event);
  }
  void ReportPort::flush() const
  {
    if (const auto state = slot->state()) state->flush();
  }
  std::optional<std::filesystem::path> ReportPort::log_file() const { return slot->log_file(); }
  int ReportPort::verbosity() const noexcept { return slot->verbosity(); }
  double ReportPort::Lambda() const noexcept { return slot->Lambda(); }
  ReportPort::operator bool() const noexcept { return slot->active(); }

  RunReporter::RunReporter(OutputPath path, const Config::OutputSettings &settings, const bool active,
                           RunReporterOptions options)
      : state(std::make_shared<internal::ReporterState>(std::move(path), settings, active, std::move(options)))
  {
  }

  RunReporter::~RunReporter() noexcept
  {
    try {
      finish();
    } catch (...) {
    }
  }

  void RunReporter::finish()
  {
    if (state) state->finish();
  }

  ReportPort RunReporter::port() const { return ReportPort(state); }
} // namespace DiFfRG
