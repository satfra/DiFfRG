// standard library
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>

// DiFfRG
#include <DiFfRG/common/csv.hh>
#include <DiFfRG/common/run_reporter.hh>

namespace DiFfRG
{
  namespace
  {
    constexpr std::string_view utf8_bom = "\xEF\xBB\xBF";

    constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();

    bool is_space(const char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f'; }

    std::string_view trim(std::string_view text)
    {
      while (!text.empty() && is_space(text.front()))
        text.remove_prefix(1);
      while (!text.empty() && is_space(text.back()))
        text.remove_suffix(1);
      return text;
    }

    /** Whether a line carries no data: empty, all whitespace, or a comment. */
    bool is_skippable(std::string_view line, const char comment)
    {
      const auto trimmed = trim(line);
      return trimmed.empty() || (comment != '\0' && trimmed.front() == comment);
    }
  } // namespace

  size_t CsvTable::n_cols() const { return columns.size(); }

  size_t CsvTable::n_rows() const { return columns.empty() ? 0 : columns.front().size(); }

  size_t CsvTable::column_index(std::string_view name) const
  {
    const auto it = std::find(header.begin(), header.end(), name);
    if (it == header.end())
      throw std::runtime_error("CsvTable::column_index: no column named '" + std::string(name) + "'.");
    return static_cast<size_t>(std::distance(header.begin(), it));
  }

  std::vector<std::string> split_csv_line(std::string_view line, const char separator)
  {
    // A trailing carriage return belongs to the line ending, not to the last field.
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
      const char c = line[i];
      if (in_quotes) {
        // Inside quotes only a quote is special: a doubled one is an escaped literal quote.
        if (c != '"')
          field.push_back(c);
        else if (i + 1 < line.size() && line[i + 1] == '"') {
          field.push_back('"');
          ++i;
        } else
          in_quotes = false;
      } else if (c == '"' && trim(field).empty()) {
        // A quote only opens a quoted field if nothing but padding precedes it.
        field.clear();
        in_quotes = true;
      } else if (c == separator) {
        fields.push_back(std::move(field));
        field.clear();
      } else
        field.push_back(c);
    }
    fields.push_back(std::move(field));

    return fields;
  }

  double parse_csv_cell(std::string_view field)
  {
    auto text = trim(field);
    if (text.empty()) return nan_value;

    // from_chars accepts a leading '-' but not a leading '+', which is a perfectly ordinary way to
    // write a positive number in a data file.
    if (text.front() == '+') {
      text.remove_prefix(1);
      if (text.empty()) return nan_value;
    }

    // Anything the parser could not consume in full is not a number, and a non-finite value is
    // normalised to NaN so that downstream NaN filtering sees every gap in the data.
    double value = nan_value;
#if defined(__cpp_lib_to_chars)
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value)) return nan_value;
#else
    // Older libc++ (e.g. AppleClang 15) has no floating-point from_chars; a classic-locale stream
    // keeps the parse locale-independent. Unlike from_chars, a stream would accept "++5".
    if (text.front() == '+') return nan_value;
    std::istringstream stream{std::string(text)};
    stream.imbue(std::locale::classic());
    stream >> value;
    if (stream.fail() || !stream.eof() || !std::isfinite(value)) return nan_value;
#endif
    return value;
  }

  CsvTable parse_csv(std::string_view content, const CsvDialect &dialect, std::string_view origin)
  {
    if (content.starts_with(utf8_bom)) content.remove_prefix(utf8_bom.size());

    CsvTable table;
    size_t n_cols = 0;
    size_t n_ragged = 0;
    bool header_pending = dialect.has_header;

    while (!content.empty()) {
      const auto break_pos = content.find('\n');
      const std::string_view line = content.substr(0, break_pos);
      content = break_pos == std::string_view::npos ? std::string_view{} : content.substr(break_pos + 1);

      if (is_skippable(line, dialect.comment)) continue;

      auto fields = split_csv_line(line, dialect.separator);

      if (header_pending) {
        header_pending = false;
        n_cols = fields.size();
        table.header.reserve(n_cols);
        for (auto &field : fields)
          table.header.emplace_back(trim(field));
        table.columns.resize(n_cols);
        continue;
      }

      if (table.columns.empty()) {
        // Without a header the first data row fixes the width.
        n_cols = fields.size();
        table.columns.resize(n_cols);
      }

      if (fields.size() != n_cols) {
        ++n_ragged;
        continue;
      }

      for (size_t i = 0; i < n_cols; ++i)
        table.columns[i].push_back(parse_csv_cell(fields[i]));
    }

    if (n_ragged > 0) {
      ReportPort report;
      report.warn("parse_csv: skipped {} row(s) whose field count did not match the {} column(s){}{}", n_ragged, n_cols,
                  origin.empty() ? "" : " in ", origin);
    }

    return table;
  }

  CsvTable read_csv(const std::string &path, const CsvDialect &dialect)
  {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("read_csv: could not open file: " + path);

    std::ostringstream buffer;
    buffer << file.rdbuf();
    if (file.bad()) throw std::runtime_error("read_csv: failed while reading file: " + path);

    return parse_csv(buffer.str(), dialect, path);
  }

  void write_csv_header(std::ostream &stream, const std::vector<std::string> &names, const CsvDialect &dialect)
  {
    // Callers that route names through strip_name never need quoting, but a name is free-form text
    // in general and must not be allowed to forge a field boundary.
    const std::string forbidden{dialect.separator, '"', '\n'};

    for (size_t i = 0; i < names.size(); ++i) {
      if (i > 0) stream << dialect.separator;

      const auto &name = names[i];
      if (name.find_first_of(forbidden) == std::string::npos)
        stream << name;
      else {
        stream << '"';
        for (const char c : name) {
          if (c == '"') stream << '"';
          stream << c;
        }
        stream << '"';
      }
    }
    stream << std::endl;
  }

  void write_csv_row(std::ostream &stream, const std::vector<double> &values, const CsvDialect &dialect)
  {
    // Number formatting is left to the stream's own flags, so callers keep control of precision.
    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0) stream << dialect.separator;
      stream << values[i];
    }
    stream << std::endl;
  }
} // namespace DiFfRG
