#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <filesystem>
#include <fstream>

#include <DiFfRG/common/config_tree.hh>

using namespace DiFfRG;

namespace
{
  const std::string toml_document = R"TOML(
# TOML allows comments, which is the whole point of supporting it.
[physical]
Lambda = 0.65
N = 2                     # an integer, not a float
model = "O(N)"

[discretization]
threads = 32
use_gpu = true

[discretization.grid]
x_grid = "0:1e-4:1.5e-2"
refine = 0

[timestepping]
final_time = 4.0
tolerances = [1e-13, 1e-7]
)TOML";

  const std::string json_document = R"JSON(
{
  "physical": { "Lambda": 0.65, "N": 2, "model": "O(N)" },
  "discretization": {
    "threads": 32,
    "use_gpu": true,
    "grid": { "x_grid": "0:1e-4:1.5e-2", "refine": 0 }
  },
  "timestepping": { "final_time": 4.0, "tolerances": [1e-13, 1e-7] }
}
)JSON";

  std::string write_temporary(const std::string &name, const std::string &content)
  {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream file(path, std::ofstream::trunc);
    file << content;
    file.close();
    return path.string();
  }
} // namespace

TEST_CASE("ConfigTree reads TOML files", "[config][common][toml]")
{
  const auto filename = write_temporary("difrg_config_tree_test.toml", toml_document);
  const ConfigTree config(filename);

  SECTION("Nested tables are addressed by json pointer")
  {
    REQUIRE(config.get_double("/physical/Lambda") == Catch::Approx(0.65));
    REQUIRE(config.get_string("/physical/model") == "O(N)");
    REQUIRE(config.get_int("/discretization/threads") == 32);
    REQUIRE(config.get_bool("/discretization/use_gpu") == true);
    REQUIRE(config.get_string("/discretization/grid/x_grid") == "0:1e-4:1.5e-2");
    REQUIRE(config.get_uint("/discretization/grid/refine") == 0u);
    REQUIRE(config.get_double("/timestepping/final_time") == Catch::Approx(4.0));
  }

  SECTION("Arrays are preserved")
  {
    const ConfigTree copy(config);
    const json::value value = copy;
    REQUIRE(value.at_pointer("/timestepping/tolerances").as_array().size() == 2);
    REQUIRE(value.at_pointer("/timestepping/tolerances/0").as_double() == Catch::Approx(1e-13));
  }

  SECTION("Defaults are returned for missing keys")
  {
    REQUIRE(config.get_double("/physical/does_not_exist", 1.5) == Catch::Approx(1.5));
    REQUIRE(config.get_string("/nope", "fallback") == "fallback");
  }
}

TEST_CASE("TOML and JSON documents with the same content compare equal", "[config][common][toml]")
{
  const auto toml_file = write_temporary("difrg_config_tree_equal.toml", toml_document);
  const auto json_file = write_temporary("difrg_config_tree_equal.json", json_document);

  REQUIRE(ConfigTree(toml_file) == ConfigTree(json_file));
  REQUIRE(ConfigTree::from_toml_string(toml_document) == ConfigTree::from_json_string(json_document));
}

TEST_CASE("ConfigTree::get_double accepts integer entries", "[config][common]")
{
  // Neither JSON nor TOML lets `2` and `2.0` be the same thing, so integers have to be
  // readable as doubles; otherwise every hand-written parameter file is a trap.
  const auto config = ConfigTree::from_toml_string("a = 2\nb = 2.0\nc = \"text\"\n");

  REQUIRE(config.get_double("/a") == Catch::Approx(2.0));
  REQUIRE(config.get_double("/b") == Catch::Approx(2.0));
  REQUIRE(config.get_double("/a", 7.0) == Catch::Approx(2.0));

  // Non-numeric entries still fail, and integer accessors stay strict.
  REQUIRE_THROWS_AS(config.get_double("/c"), std::runtime_error);
  REQUIRE_THROWS_AS(config.get_int("/b"), std::runtime_error);
  REQUIRE(config.get_double("/c", 3.0) == Catch::Approx(3.0));
}

TEST_CASE("ConfigTree integer getters accept values stored as unsigned", "[config][common]")
{
  // boost::json keeps a C++ `unsigned` in its uint64 arm and an `int` in its int64 arm, and
  // as_int64() throws on the uint64 arm. get_uint -- the accessor that exists precisely for
  // unsigned values -- therefore used to be unable to read one, and get_int with it. A config
  // parsed from a file never hit this; one built in C++ from unsigned values always did, and it
  // failed quietly, because get_uint_or_warn catches the exception and returns its default. The
  // caller then silently ran with the default instead of the value it had set.
  const ConfigTree config =
      boost::json::value({{"unsigned_value", 4u}, {"signed_value", 4}, {"negative", -1}, {"text", "no"}});

  REQUIRE(config.get_uint("/unsigned_value") == 4u);
  REQUIRE(config.get_uint("/signed_value") == 4u);
  REQUIRE(config.get_int("/unsigned_value") == 4);

  // The fallback getters must see the value, not their default -- this is the quiet failure.
  REQUIRE(config.get_uint_or_warn("/unsigned_value", 99u) == 4u);
  REQUIRE(config.get_int_or_warn("/unsigned_value", 99) == 4);

  // Widening the accepted arm must not weaken the existing checks.
  REQUIRE_THROWS_AS(config.get_uint("/negative"), std::runtime_error);
  REQUIRE_THROWS_AS(config.get_uint("/text"), std::runtime_error);
  REQUIRE_THROWS_AS(config.get_int("/text"), std::runtime_error);
}

TEST_CASE("ConfigTree writes TOML that reads back identically", "[config][common][toml]")
{
  const auto config = ConfigTree::from_json_string(json_document);

  std::stringstream stream;
  config.print_toml(stream);
  const auto written = stream.str();

  INFO("written TOML:\n" << written);
  REQUIRE(ConfigTree::from_toml_string(written) == config);

  SECTION("Empty sections and empty arrays survive the round trip")
  {
    // An empty section is what --generate-parameter-file writes for /physical. Written as
    // a bare [[physical]] it would read back as an array holding one table.
    const auto sparse = ConfigTree::from_json_string(R"({"physical": {}, "empty": [], "a": 1})");

    std::stringstream out;
    sparse.print_toml(out);
    INFO("written TOML:\n" << out.str());

    const auto reread = ConfigTree::from_toml_string(out.str());
    const json::value value = reread;
    REQUIRE(value.at_pointer("/physical").is_object());
    REQUIRE(value.at_pointer("/physical").as_object().empty());
    REQUIRE(value.at_pointer("/empty").is_array());
    REQUIRE(value.at_pointer("/empty").as_array().empty());
    REQUIRE(reread == sparse);
  }

  SECTION("Full double precision survives the round trip")
  {
    const auto precise = ConfigTree::from_toml_string("a = 1.2345678901234567e-13\nb = 4.0\n");
    std::stringstream out;
    precise.print_toml(out);
    const auto reread = ConfigTree::from_toml_string(out.str());

    const json::value original = precise;
    const json::value copy = reread;
    REQUIRE(copy.at_pointer("/a").as_double() == original.at_pointer("/a").as_double());
    REQUIRE(copy.at_pointer("/b").as_double() == original.at_pointer("/b").as_double());
  }
}

TEST_CASE("ConfigTree reports file and syntax errors", "[config][common][toml]")
{
  SECTION("Missing file")
  {
    REQUIRE_THROWS_AS(ConfigTree("this_file_does_not_exist.toml"), std::runtime_error);
    REQUIRE_THROWS_AS(ConfigTree("this_file_does_not_exist.json"), std::runtime_error);
  }

  SECTION("Malformed TOML names the file")
  {
    const auto filename = write_temporary("difrg_config_tree_broken.toml", "this is = = not toml\n");
    REQUIRE_THROWS_WITH(ConfigTree(filename), Catch::Matchers::ContainsSubstring(filename));
  }

  SECTION("Malformed TOML string")
  {
    REQUIRE_THROWS_AS(ConfigTree::from_toml_string("[unterminated\n"), std::runtime_error);
  }
}

TEST_CASE("The file format is chosen by extension", "[config][common][toml]")
{
  REQUIRE(internal::has_toml_extension("parameter.toml"));
  REQUIRE(internal::has_toml_extension("parameter.TOML"));
  REQUIRE(internal::has_toml_extension("some/path/parameter.tml"));
  REQUIRE_FALSE(internal::has_toml_extension("parameter.json"));
  REQUIRE_FALSE(internal::has_toml_extension("parameter"));

  // A file without an extension keeps the historical behaviour of being read as JSON.
  const auto filename = write_temporary("difrg_config_tree_no_extension", json_document);
  REQUIRE(ConfigTree(filename) == ConfigTree::from_json_string(json_document));
}

TEST_CASE("Warning getters return the value when present and the default when not", "[config][common][defaults]")
{
  const ConfigTree config = ConfigTree::from_json_string(R"JSON({
    "physical": {"Lambda": 0.65},
    "discretization": {"fe_order": 4},
    "integration": {"x_order": 48},
    "output": {"name": "run", "vtk": false}
  })JSON");

  // Present keys behave exactly like the plain getters and must not warn.
  CHECK(config.get_double_or_warn("/physical/Lambda", 1.0) == Catch::Approx(0.65));
  CHECK(config.get_uint_or_warn("/discretization/fe_order", 3) == 4);
  CHECK(config.get_int_or_warn("/integration/x_order", 32) == 48);
  CHECK(config.get_string_or_warn("/output/name", "output") == "run");
  CHECK(config.get_bool_or_warn("/output/vtk", true) == false);

  // Absent keys fall back to the default (and print one warning per key to stderr).
  CHECK(config.get_double_or_warn("/physical/T", 0.1) == Catch::Approx(0.1));
  CHECK(config.get_uint_or_warn("/discretization/overintegration", 8) == 8);
  CHECK(config.get_int_or_warn("/output/verbosity", 0) == 0);
  CHECK(config.get_string_or_warn("/output/folder", "./") == "./");
  CHECK(config.get_bool_or_warn("/output/hdf5", true) == true);

  // Repeated reads of the same missing key stay consistent (the warning itself is once-only).
  CHECK(config.get_double_or_warn("/physical/T", 0.1) == Catch::Approx(0.1));
}
