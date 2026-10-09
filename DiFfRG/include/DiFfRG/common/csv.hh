#pragma once

// standard library
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief The CSV dialect DiFfRG reads and writes.
   *
   * The same rules are implemented by the analysis-side readers in julia/src/io/csv.jl and
   * python/DiFfRG/file_io/csv.py, so a file written by CsvOutput loads identically in all three.
   *
   * Data cells hold numbers only and are parsed with std::from_chars, which is locale-independent.
   * A cell that is empty or does not parse becomes NaN rather than raising, so a run that wrote an
   * inf or left a gap still loads and the gap stays visible in the data. Header names may be quoted
   * in the RFC-4180 sense, which is the only way a name can contain the separator. Rows whose field
   * count differs from the header are skipped. Comment lines, blank lines, a UTF-8 BOM and CRLF
   * line endings are all tolerated.
   */
  struct CsvDialect {
    /** The character separating fields within a line. */
    char separator = ',';
    /** Whether the first non-comment, non-blank line names the columns. */
    bool has_header = false;
    /** Lines whose first non-space character is this are skipped. Set to '\0' to disable. */
    char comment = '#';
  };

  /**
   * @brief A numeric table, stored one vector per column.
   *
   * Column-major because every consumer wants whole columns: the interpolators build one
   * interpolant per column, and indexed cell access is a two-step lookup either way.
   */
  struct CsvTable {
    /** Column names, or empty if the file carried no header. */
    std::vector<std::string> header;
    /** One vector of values per column. All columns have the same length. */
    std::vector<std::vector<double>> columns;

    size_t n_rows() const;
    size_t n_cols() const;

    /**
     * @brief Look up a column by name.
     *
     * Unlike an unparseable cell, an unknown column name is a programming error rather than bad
     * data, so this throws instead of yielding NaN.
     *
     * @throws std::runtime_error if the table has no header or no column of that name.
     */
    size_t column_index(std::string_view name) const;
  };

  /**
   * @brief Read a CSV file into a numeric table.
   *
   * @param path The file to read.
   * @param dialect The separator, header and comment conventions to apply.
   * @throws std::runtime_error if the file cannot be opened.
   */
  CsvTable read_csv(const std::string &path, const CsvDialect &dialect = {});

  /**
   * @brief Parse CSV held in memory into a numeric table.
   *
   * @param content The CSV text.
   * @param dialect The separator, header and comment conventions to apply.
   * @param origin A name for the source, used only to decorate diagnostics.
   */
  CsvTable parse_csv(std::string_view content, const CsvDialect &dialect = {}, std::string_view origin = {});

  /**
   * @brief Split one CSV line into its fields, honouring quoting.
   *
   * Exposed because it is the one place that defines what a field boundary is; both the reader and
   * the header path use it.
   */
  std::vector<std::string> split_csv_line(std::string_view line, char separator);

  /**
   * @brief Parse a single numeric cell, yielding NaN for anything that is not a number.
   */
  double parse_csv_cell(std::string_view field);

  /**
   * @brief Write a header row, quoting any name that would otherwise forge a field boundary.
   */
  void write_csv_header(std::ostream &stream, const std::vector<std::string> &names, const CsvDialect &dialect = {});

  /**
   * @brief Write one row of values.
   */
  void write_csv_row(std::ostream &stream, const std::vector<double> &values, const CsvDialect &dialect = {});
} // namespace DiFfRG
