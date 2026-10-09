#pragma once

// standard library
#include <cstddef>
#include <string>

// DiFfRG
#include <DiFfRG/common/csv.hh>

namespace DiFfRG
{
  /**
   * @brief This class reads a .csv file and allows to access the data.
   *
   * Follows the dialect described in CsvDialect: cells that are not numbers read back as NaN, and
   * '#' comment lines are skipped.
   */
  class CSVReader
  {
  public:
    /**
     * @brief Construct a new External Data Interpolator object
     *
     * @param input_file The .csv file to read the data from.
     * @param separator The separator used in the .csv file.
     * @param has_header Whether the .csv file has a header.
     */
    CSVReader(std::string input_file, char separator = ',', bool has_header = false);

    /**
     * @brief Get the number of rows in the .csv file.
     *
     * @return uint The number of rows.
     */
    size_t n_rows() const;

    /**
     * @brief Get the number of columns in the .csv file.
     *
     * @return uint The number of columns.
     */
    size_t n_cols() const;

    /**
     * @brief Get the stored value at a given row and column.
     *
     * @param col The name of the column from which to get the value.
     * @param row The row from which to get the value.
     * @return double The data.
     */
    double value(const std::string &col, const size_t row) const;

    /**
     * @brief Get the stored value at a given row and column.
     *
     * @param col The column from which to get the value.
     * @param row The row from which to get the value.
     * @return double The data.
     */
    double value(const size_t col, const size_t row) const;

  private:
    std::string input_file;
    CsvTable table;
  };
} // namespace DiFfRG
