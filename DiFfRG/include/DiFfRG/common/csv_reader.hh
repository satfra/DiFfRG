#pragma once

// standard library
#include <cstddef>
#include <string>

// DiFfRG
#include <DiFfRG/common/csv.hh>

namespace DiFfRG
{
  /**
   * @brief Reads a numeric .csv file and gives access to its cells.
   *
   * Follows the rules of CsvDialect: cells that are not finite numbers read as NaN, and lines starting with `#`
   * are skipped.
   */
  class CSVReader
  {
  public:
    /**
     * @brief Read a .csv file.
     *
     * @param input_file The file to read.
     * @param separator The field separator.
     * @param has_header Whether the first line names the columns.
     * @throws std::runtime_error if the file cannot be opened.
     */
    CSVReader(std::string input_file, char separator = ',', bool has_header = false);

    /// Number of data rows (header, comments and skipped rows not counted).
    size_t n_rows() const;

    /// Number of columns.
    size_t n_cols() const;

    /**
     * @brief The value in column `col` and data row `row`.
     *
     * @param col The column name, as given in the header.
     * @param row The data row, counted from 0.
     * @throws std::runtime_error if the file has no header, no such column, or `row` is out of range.
     */
    double value(const std::string &col, const size_t row) const;

    /**
     * @brief The value in column `col` and data row `row`.
     *
     * @param col The column, counted from 0.
     * @param row The data row, counted from 0.
     * @throws std::runtime_error if `col` or `row` is out of range.
     */
    double value(const size_t col, const size_t row) const;

  private:
    std::string input_file;
    CsvTable table;
  };
} // namespace DiFfRG
