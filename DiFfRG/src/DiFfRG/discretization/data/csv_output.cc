// standard library
#include <filesystem>

// DiFfRG
#include <DiFfRG/common/csv.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/csv_output.hh>

namespace DiFfRG
{
  CsvOutput::CsvOutput(const std::string top_folder, const std::string output_name)
      : top_folder(make_folder(top_folder)), output_name(output_name), Lambda(-1.)
  {
    create_folder(this->top_folder);
    const std::filesystem::path path = std::filesystem::path(this->top_folder) / this->output_name;
    create_folder(path.parent_path().string());

    output_stream = std::ofstream(path, std::ofstream::trunc);
    if (!output_stream) throw std::runtime_error("CsvOutput: could not open '" + top_folder + output_name + "'.");
    output_stream << std::scientific;
  }

  void CsvOutput::value(const std::string &name, const double value)
  {
    if (!written_this_frame.insert(name).second)
      throw std::runtime_error("CsvOutput::value: The field '" + name + "' was written twice in one frame.");
    if ((time_values.size() == 0) &&
        std::find(insertion_order.begin(), insertion_order.end(), name) == insertion_order.end())
      insertion_order.push_back(name);
    values[name].push_back(value);
  }

  void CsvOutput::validate_frame() const
  {
    const auto expected_field_count = header.empty() ? insertion_order.size() : header.size() - 1 - int(Lambda > 0);
    if (written_this_frame.size() != expected_field_count)
      throw std::runtime_error("CsvOutput::flush: The frame for '" + output_name + "' is incomplete.");

    if (!header.empty())
      for (const auto &name : insertion_order)
        if (!written_this_frame.contains(name))
          throw std::runtime_error("CsvOutput::flush: Missing field '" + name + "' in '" + output_name + "'.");
  }

  void CsvOutput::flush(const double time)
  {
    validate_frame();

    if (time_values.size() == 0) {
      // Create the header.
      header.clear();
      header.push_back("t");
      if (Lambda > 0) header.push_back("k [GeV]");
      for (const auto &entry : insertion_order)
        header.push_back(entry);

      // strip_name keeps the written names free of separators and quotes.
      std::vector<std::string> names;
      names.reserve(header.size());
      for (const auto &entry : header)
        names.push_back(strip_name(entry));
      write_csv_header(output_stream, names);
    }

    time_values.push_back(time);
    k_values.push_back(std::exp(-time) * Lambda);

    // Collect this frame's values in header order and write them as one row.
    std::vector<double> row;
    row.reserve(header.size());
    row.push_back(time);
    if (Lambda > 0) row.push_back(k_values.back());
    for (const auto &entry : header) {
      if (entry == "t" || entry == "k [GeV]") continue;
      row.push_back(values[entry].back());
    }
    write_csv_row(output_stream, row);
    if (!output_stream) throw std::runtime_error("CsvOutput::flush: write failed for '" + output_name + "'.");
    written_this_frame.clear();
  }

  void CsvOutput::set_Lambda(const double Lambda)
  {
    // This needs safety checks, so that Lambda can only be set once, before any call to flush().
    if (time_values.size() > 0 && !is_close(this->Lambda, Lambda))
      throw std::runtime_error("Lambda has either already been set or there has been an attempt to change it.");
    this->Lambda = Lambda;
  }
} // namespace DiFfRG
