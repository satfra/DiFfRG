#include <catch2/catch_all.hpp>

#include <DiFfRG/common/csv.hh>
#include <DiFfRG/common/csv_reader.hh>
#include <DiFfRG/common/external_data_interpolator.hh>
#include <DiFfRG/discretization/data/csv_output.hh>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

using namespace DiFfRG;

namespace
{
  /** A file under the system temp directory, removed again when the test leaves scope. */
  class ScratchFile
  {
  public:
    ScratchFile(const std::string &suffix, const std::string &content)
    {
      static std::mt19937_64 rng(std::random_device{}());
      path = std::filesystem::temp_directory_path() /
             ("diffrg_csv_" + std::to_string(rng()) + "_" + suffix);
      std::ofstream file(path, std::ios::binary);
      file << content;
    }
    ~ScratchFile() { std::error_code ec; std::filesystem::remove(path, ec); }

    ScratchFile(const ScratchFile &) = delete;
    ScratchFile &operator=(const ScratchFile &) = delete;

    std::string name() const { return path.string(); }

  private:
    std::filesystem::path path;
  };
} // namespace

TEST_CASE("parse_csv reads headerless numeric data", "[common][csv]")
{
  SECTION("a single column, as the external-data files are shaped")
  {
    const auto table = parse_csv("2.000000000000000000e+01\n1.980099999999999838e+01\n");
    REQUIRE(table.n_cols() == 1);
    REQUIRE(table.n_rows() == 2);
    CHECK(table.columns[0][0] == Catch::Approx(20.0));
    CHECK(table.columns[0][1] == Catch::Approx(19.801));
    CHECK(table.header.empty());
  }

  SECTION("several columns")
  {
    const auto table = parse_csv("1,2,3\n4,5,6\n");
    REQUIRE(table.n_cols() == 3);
    REQUIRE(table.n_rows() == 2);
    CHECK(table.columns[2][1] == Catch::Approx(6.0));
  }
}

TEST_CASE("parse_csv reads a header", "[common][csv]")
{
  const auto table = parse_csv("t,kGeV,m2\n0.0,20.0,1.5\n0.1,19.8,1.4\n", {',', true});

  REQUIRE(table.header == std::vector<std::string>{"t", "kGeV", "m2"});
  REQUIRE(table.n_rows() == 2);
  CHECK(table.columns[table.column_index("m2")][0] == Catch::Approx(1.5));
  CHECK_THROWS_AS(table.column_index("not_a_column"), std::runtime_error);
}

TEST_CASE("parse_csv yields NaN rather than throwing on bad cells", "[common][csv]")
{
  // Blank, non-numeric and non-finite cells all collapse to NaN, so that every gap in the data
  // looks the same to the NaN filtering downstream.
  const auto table = parse_csv("1.0,,garbage,inf,-inf,nan,2.5e\n");

  REQUIRE(table.n_cols() == 7);
  CHECK(table.columns[0][0] == Catch::Approx(1.0));
  for (size_t col = 1; col < table.n_cols(); ++col) {
    INFO("column " << col);
    CHECK(std::isnan(table.columns[col][0]));
  }
}

TEST_CASE("parse_csv tolerates the usual file-format noise", "[common][csv]")
{
  SECTION("comment lines and blank lines are skipped")
  {
    const auto table = parse_csv("# a leading comment\n\n1,2\n   \n# another\n3,4\n");
    REQUIRE(table.n_rows() == 2);
    CHECK(table.columns[1][1] == Catch::Approx(4.0));
  }

  SECTION("comment skipping can be switched off")
  {
    const auto table = parse_csv("#1,2\n3,4\n", {',', false, '\0'});
    REQUIRE(table.n_rows() == 2);
    CHECK(std::isnan(table.columns[0][0]));
  }

  SECTION("CRLF line endings")
  {
    const auto table = parse_csv("1,2\r\n3,4\r\n");
    REQUIRE(table.n_rows() == 2);
    CHECK(table.columns[1][0] == Catch::Approx(2.0));
    CHECK(table.columns[1][1] == Catch::Approx(4.0));
  }

  SECTION("a UTF-8 BOM")
  {
    const auto table = parse_csv("\xEF\xBB\xBF" "t,x\n1,2\n", {',', true});
    REQUIRE(table.header == std::vector<std::string>{"t", "x"});
    CHECK(table.columns[0][0] == Catch::Approx(1.0));
  }

  SECTION("a missing final newline")
  {
    const auto table = parse_csv("1,2\n3,4");
    REQUIRE(table.n_rows() == 2);
  }

  SECTION("whitespace padding around cells")
  {
    const auto table = parse_csv("  1.5 ,\t2.5\n");
    CHECK(table.columns[0][0] == Catch::Approx(1.5));
    CHECK(table.columns[1][0] == Catch::Approx(2.5));
  }
}

TEST_CASE("parse_csv honours quoted header names", "[common][csv]")
{
  // Quoting is the only way a column name can contain the separator.
  const auto table = parse_csv("\"k, in GeV\",\"say \"\"hi\"\"\"\n1,2\n", {',', true});

  REQUIRE(table.header.size() == 2);
  CHECK(table.header[0] == "k, in GeV");
  CHECK(table.header[1] == "say \"hi\"");
  CHECK(table.columns[0][0] == Catch::Approx(1.0));
}

TEST_CASE("parse_csv skips rows of the wrong width", "[common][csv]")
{
  const auto table = parse_csv("a,b\n1,2\n3\n4,5,6\n7,8\n", {',', true});

  REQUIRE(table.n_cols() == 2);
  REQUIRE(table.n_rows() == 2);
  CHECK(table.columns[0][0] == Catch::Approx(1.0));
  CHECK(table.columns[0][1] == Catch::Approx(7.0));
}

TEST_CASE("parse_csv accepts a non-comma separator", "[common][csv]")
{
  const auto table = parse_csv("1;2\n3;4\n", {';'});
  REQUIRE(table.n_cols() == 2);
  CHECK(table.columns[1][1] == Catch::Approx(4.0));
}

TEST_CASE("parse_csv is independent of the decimal locale", "[common][csv]")
{
  // std::from_chars ignores the global locale, unlike the stringstream conversion it replaced.
  const auto table = parse_csv("1.25,2.5\n");
  CHECK(table.columns[0][0] == Catch::Approx(1.25));
  CHECK(table.columns[1][0] == Catch::Approx(2.5));
}

TEST_CASE("parse_csv accepts the sign and exponent spellings found in data files", "[common][csv]")
{
  // A leading '+' needs handling of its own: std::from_chars rejects it outright.
  const auto table = parse_csv("+1.5,-2.5,+1.0e-3,1.797e+308\n");
  CHECK(table.columns[0][0] == Catch::Approx(1.5));
  CHECK(table.columns[1][0] == Catch::Approx(-2.5));
  CHECK(table.columns[2][0] == Catch::Approx(1.0e-3));
  CHECK(table.columns[3][0] == Catch::Approx(1.797e308));
}

TEST_CASE("read_csv reports a file it cannot open", "[common][csv]")
{
  const std::string missing = (std::filesystem::temp_directory_path() / "diffrg_no_such_file.csv").string();
  REQUIRE_THROWS_WITH(read_csv(missing), Catch::Matchers::ContainsSubstring(missing));
}

TEST_CASE("CSVReader exposes the table by index and by name", "[common][csv]")
{
  const ScratchFile file("reader.csv", "t,m2\n0.0,1.5\n0.1,1.25\n");
  const CSVReader reader(file.name(), ',', true);

  REQUIRE(reader.n_cols() == 2);
  REQUIRE(reader.n_rows() == 2);
  CHECK(reader.value(1, 0) == Catch::Approx(1.5));
  CHECK(reader.value("m2", 1) == Catch::Approx(1.25));
  CHECK_THROWS_AS(reader.value("absent", 0), std::runtime_error);
  CHECK_THROWS_AS(reader.value(size_t(5), 0), std::runtime_error);
  CHECK_THROWS_AS(reader.value(size_t(0), 99), std::runtime_error);
}

TEST_CASE("ExternalDataInterpolator reads and interpolates a data file", "[common][csv]")
{
  // A linear column, which the order-1 barycentric rational reproduces exactly.
  const ScratchFile file("linear.csv", "0,0\n1,2\n2,4\n3,6\n4,8\n5,10\n");
  const ExternalDataInterpolator interpolator({file.name()});

  CHECK(interpolator.value(2.5, 1) == Catch::Approx(5.0));
  CHECK(interpolator.derivative(2.5, 1) == Catch::Approx(2.0));

  SECTION("values outside the data range clamp to the closest boundary")
  {
    CHECK(interpolator.value(-10.0, 1) == Catch::Approx(0.0));
    CHECK(interpolator.value(100.0, 1) == Catch::Approx(10.0));
  }
}

TEST_CASE("ExternalDataInterpolator drops rows the parser could not read", "[common][csv]")
{
  // The blank cell becomes NaN in the parser, and the interpolator then drops that whole row --
  // this is the coupling between the two that keeps a gap in a data file from poisoning the fit.
  const ScratchFile file("gap.csv", "0,0\n1,2\n2,\n3,6\n4,8\n5,10\n");
  const ExternalDataInterpolator interpolator({file.name()});

  CHECK(interpolator.value(2.5, 1) == Catch::Approx(5.0));
  CHECK(interpolator.value(1.0, 1) == Catch::Approx(2.0));
}

TEST_CASE("ExternalDataInterpolator concatenates several files and post-processes them", "[common][csv]")
{
  const ScratchFile x_file("x.csv", "0\n1\n2\n3\n4\n5\n");
  const ScratchFile y_file("y.csv", "0\n1\n2\n3\n4\n5\n");

  // The second column is doubled on read-in, so the two columns must come out different.
  const ExternalDataInterpolator interpolator({x_file.name(), y_file.name()},
                                              {[](double v) { return v; }, [](double v) { return 2. * v; }});

  CHECK(interpolator.value(2.5, 0) == Catch::Approx(2.5));
  CHECK(interpolator.value(2.5, 1) == Catch::Approx(5.0));
}

TEST_CASE("ExternalDataInterpolator honours a non-zero x_column", "[common][csv]")
{
  // Column 1 is the independent variable here. Consistency checking used to evaluate against
  // column 0 regardless, which made this combination fail for data that is perfectly fine.
  const ScratchFile file("xcol.csv", "0,0\n2,1\n4,2\n6,3\n8,4\n10,5\n");
  const ExternalDataInterpolator interpolator({file.name()}, ',', false, 1);

  CHECK(interpolator.value(2.5, 0) == Catch::Approx(5.0));
  CHECK(interpolator.value(2.5, 1) == Catch::Approx(2.5));
}

TEST_CASE("ExternalDataInterpolator reads a file with a header", "[common][csv]")
{
  const ScratchFile file("header.csv", "# a comment above the header\nk,eta\n0,0\n1,2\n2,4\n3,6\n4,8\n");
  const ExternalDataInterpolator interpolator({file.name()}, ',', true);

  CHECK(interpolator.value(2.5, 1) == Catch::Approx(5.0));
}

TEST_CASE("CsvOutput and the reader agree on the format", "[common][csv]")
{
  // The round trip is what pins the writer and the reader to one dialect.
  const auto folder = std::filesystem::temp_directory_path() / "diffrg_csv_roundtrip";
  std::filesystem::remove_all(folder);

  const std::vector<double> times{0.0, 0.5, 1.0};
  const std::vector<double> masses{1.5, 1.25, 1.125};
  {
    CsvOutput out(folder.string(), "run");
    for (size_t i = 0; i < times.size(); ++i) {
      // A name carrying characters strip_name removes, to check the header survives the round trip.
      out.value("m2 [GeV^2]", masses[i]);
      out.flush(times[i]);
    }
  }

  const auto written = folder / "run";
  REQUIRE(std::filesystem::exists(written));

  const auto table = read_csv(written.string(), {',', true});
  REQUIRE(table.header == std::vector<std::string>{"t", "m2GeV2"});
  REQUIRE(table.n_rows() == times.size());
  for (size_t i = 0; i < times.size(); ++i) {
    INFO("row " << i);
    CHECK(table.columns[0][i] == Catch::Approx(times[i]));
    CHECK(table.columns[1][i] == Catch::Approx(masses[i]));
  }

  std::filesystem::remove_all(folder);
}
