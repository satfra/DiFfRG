#include <DiFfRG/discretization/data/diagnostic_port.hh>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/csv_output.hh>

#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

namespace DiFfRG
{
  struct DiagnosticPort::State {
    State(const OutputPath &path, const double Lambda, const bool active)
        : root(path.root()), output_name(path.run_name()), Lambda(Lambda), active(active)
    {
    }

    CsvOutput &table(const std::string &name)
    {
      auto found = tables.find(name);
      if (found != tables.end()) return found->second;

      const std::string filename =
          output_name + "_" + OutputPath::checked_relative(name, "diagnostic table").generic_string();
      auto [it, inserted] = tables.emplace(name, CsvOutput(root.string(), filename));
      if (inserted) it->second.set_Lambda(Lambda);
      return it->second;
    }

    std::mutex mutex;
    std::filesystem::path root;
    std::string output_name;
    double Lambda = -1.;
    bool active = true;
    std::map<std::string, CsvOutput> tables;
    std::exception_ptr terminal_error;
  };

  DiagnosticPort DiagnosticPort::create(const OutputPath &path, const double Lambda, const bool active)
  {
    return DiagnosticPort(std::make_shared<State>(path, Lambda, active));
  }

  void DiagnosticPort::record(const std::string &table, const double time, const Record &fields) const
  {
    if (!state || !state->active) return;
    if (table.empty()) throw std::invalid_argument("DiagnosticPort::record: table name must not be empty.");
    if (fields.empty())
      throw std::invalid_argument("DiagnosticPort::record: a record must contain at least one field.");

    std::unordered_set<std::string> field_names;
    for (const auto &[name, value] : fields) {
      if (name.empty()) throw std::invalid_argument("DiagnosticPort::record: field names must not be empty.");
      if (!field_names.insert(name).second)
        throw std::invalid_argument("DiagnosticPort::record: duplicate field '" + name + "'.");
    }

    std::scoped_lock lock(state->mutex);
    if (state->terminal_error) std::rethrow_exception(state->terminal_error);
    try {
      auto &sink = state->table(table);
      for (const auto &[name, value] : fields)
        sink.value(name, value);
      sink.flush(time);
    } catch (...) {
      state->terminal_error = std::current_exception();
      throw;
    }
  }

  void DiagnosticPort::record(const std::string &table, const double time,
                              const std::initializer_list<Field> fields) const
  {
    record(table, time, Record(fields));
  }

  void DiagnosticPort::scalar(const std::string &table, const std::string &name, const double time,
                              const double value) const
  {
    record(table, time, {{name, value}});
  }

  void DiagnosticPort::write_text(const std::string &relative_path, const std::string &contents) const
  {
    if (!state || !state->active) return;
    std::scoped_lock lock(state->mutex);
    if (state->terminal_error) std::rethrow_exception(state->terminal_error);
    const auto target = state->root / OutputPath::checked_relative(relative_path, "text artifact");
    create_folder(target.parent_path().string());
    const auto temporary = target.string() + ".tmp";
    {
      std::ofstream stream(temporary, std::ofstream::binary | std::ofstream::trunc);
      if (!stream) throw std::runtime_error("DiagnosticPort::write_text: could not open '" + temporary + "'.");
      stream << contents;
      if (!stream) throw std::runtime_error("DiagnosticPort::write_text: write failed for '" + temporary + "'.");
    }
    std::filesystem::rename(temporary, target);
  }
} // namespace DiFfRG
