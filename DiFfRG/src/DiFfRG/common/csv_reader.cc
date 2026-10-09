// standard library
#include <stdexcept>

// DiFfRG
#include <DiFfRG/common/csv_reader.hh>

namespace DiFfRG
{
  CSVReader::CSVReader(std::string input_file, char separator, bool has_header)
      : input_file(std::move(input_file)), table(read_csv(this->input_file, {separator, has_header}))
  {
  }

  double CSVReader::value(const size_t col, const size_t row) const
  {
    if (col >= table.n_cols() || row >= table.n_rows())
      throw std::runtime_error("CSVReader::value: (" + std::to_string(col) + ", " + std::to_string(row) +
                               ") is outside the " + std::to_string(table.n_cols()) + "x" +
                               std::to_string(table.n_rows()) + " table read from '" + input_file + "'.");
    return table.columns[col][row];
  }

  double CSVReader::value(const std::string &col, const size_t row) const
  {
    return value(table.column_index(col), row);
  }

  size_t CSVReader::n_rows() const { return table.n_rows(); }

  size_t CSVReader::n_cols() const { return table.n_cols(); }

} // namespace DiFfRG
