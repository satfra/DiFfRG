#pragma once

#include <DiFfRG/discretization/data/output_path.hh>

#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace DiFfRG
{
  /**
   * A small, copyable handle for independent scalar diagnostics.
   *
   * The handle owns no global state. Copies share the run-owned sink state and may
   * safely be used from different threads. Each record is appended atomically to
   * one logical CSV table.
   */
  class DiagnosticPort
  {
  public:
    using Field = std::pair<std::string, double>;
    using Record = std::vector<Field>;

    DiagnosticPort() = default;

    void record(const std::string &table, double time, const Record &fields) const;
    void record(const std::string &table, double time, std::initializer_list<Field> fields) const;
    void scalar(const std::string &table, const std::string &name, double time, double value) const;
    void write_text(const std::string &relative_path, const std::string &contents) const;

    explicit operator bool() const noexcept { return static_cast<bool>(state); }

  private:
    struct State;
    explicit DiagnosticPort(std::shared_ptr<State> state) : state(std::move(state)) {}

    static DiagnosticPort create(const OutputPath &path, double Lambda, bool active);

    std::shared_ptr<State> state;

    template <unsigned int, typename> friend class OutputSession_impl;
  };
} // namespace DiFfRG
