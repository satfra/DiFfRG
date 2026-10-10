#pragma once

// standard library
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file csv.hh
 * @brief Reading and writing numeric CSV files.
 */

namespace DiFfRG
{
  /**
   * @brief The CSV dialect DiFfRG reads and writes.
   *
   * The Python reader python/DiFfRG/file_io/csv.py implements the same rules.
   *
   * - Data cells hold numbers only; parsing does not depend on the locale.
   * - A cell that is empty, does not parse, or is infinite reads as NaN, so gaps stay visible in the data.
   *   In particular `inf` and `-inf` are read back as NaN.
   * - Header names may be quoted with `"` (a quote inside is written `""`), which is the only way a name can
   *   contain the separator. A quoted name cannot span several lines.
   * - Rows whose number of fields differs from the first row (the header, if there is one) are skipped with a
   *   warning.
   * - Blank lines, lines whose first non-space character is the comment character, a UTF-8 byte order mark
   *   and Windows line endings are tolerated.
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

    /// Number of data rows (header, comments and skipped rows not counted).
    size_t n_rows() const;
    /// Number of columns.
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
   * @param line One line, without the line break.
   * @param separator The field separator.
   */
  std::vector<std::string> split_csv_line(std::string_view line, char separator);

  /**
   * @brief Parse a single numeric cell; NaN for anything that is not a finite number.
   *
   * @param field The cell text; surrounding whitespace and a leading `+` are allowed.
   */
  double parse_csv_cell(std::string_view field);

  /**
   * @brief Write a header row. Names containing the separator, `"` or a line break are quoted.
   *
   * @param stream The stream to write to.
   * @param names The column names.
   * @param dialect Provides the separator.
   */
  void write_csv_header(std::ostream &stream, const std::vector<std::string> &names, const CsvDialect &dialect = {});

  /**
   * @brief Write one row of values, followed by a line break.
   *
   * The number format (precision) is that of `stream`.
   *
   * @param stream The stream to write to.
   * @param values The values.
   * @param dialect Provides the separator.
   */
  void write_csv_row(std::ostream &stream, const std::vector<double> &values, const CsvDialect &dialect = {});
} // namespace DiFfRG
