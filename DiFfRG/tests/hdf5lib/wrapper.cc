#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <hdf5lib/hdf5.hh>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace
{
  /// Per-test scratch path; auto-deleted on scope exit.
  struct ScratchFile {
    std::filesystem::path path;
    ScratchFile(const std::string &tag) : path(make_path(tag)) { remove(); }
    ~ScratchFile() { remove(); }
    void remove() noexcept
    {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
    std::string str() const { return path.string(); }
    static std::filesystem::path make_path(const std::string &tag)
    {
      static std::mt19937_64 rng{std::random_device{}()};
      auto suffix = std::to_string(rng());
      return std::filesystem::temp_directory_path() / ("DiFfRG_hdf5lib_" + tag + "_" + suffix + ".h5");
    }
  };
} // namespace

using namespace DiFfRG::hdf5;

TEST_CASE("File lifecycle", "[hdf5lib][file]")
{
  ScratchFile sf("file_lifecycle");

  {
    File f = File::open(sf.str(), Access::Truncate);
    REQUIRE(f.is_open());
    f.close();
    REQUIRE_FALSE(f.is_open());
  }
  REQUIRE(std::filesystem::exists(sf.path));
  {
    File f = File::open(sf.str(), Access::ReadOnly);
    REQUIRE(f.is_open());
  }

  ScratchFile missing("file_missing");
  REQUIRE_THROWS_AS(File::open(missing.str(), Access::ReadOnly), std::runtime_error);
}

TEST_CASE("Move and copy semantics for handles", "[hdf5lib][handle]")
{
  ScratchFile sf("handle_semantics");
  File f = File::open(sf.str(), Access::Truncate);
  Group root = f.root();
  Group g = root.create_group("g");

  // Copy a Group: both handles must remain valid.
  Group g_copy = g;
  REQUIRE(g.valid());
  REQUIRE(g_copy.valid());
  // Both should see the same subgroup we create through one of them.
  g.create_group("inner");
  REQUIRE(g_copy.has_group("inner"));

  // Move a Group: source becomes invalid, destination owns the handle.
  Group g_moved = std::move(g_copy);
  REQUIRE(g_moved.valid());
  REQUIRE_FALSE(g_copy.valid());
  REQUIRE(g_moved.has_group("inner"));

  // Same for File.
  File f_moved = std::move(f);
  REQUIRE(f_moved.is_open());
  REQUIRE_FALSE(f.is_open());
}

TEST_CASE("Nested groups", "[hdf5lib][group]")
{
  ScratchFile sf("nested_groups");
  File f = File::open(sf.str(), Access::Truncate);
  Group root = f.root();

  Group a = root.create_group("a");
  Group b = a.create_group("b");
  Group c = b.create_group("c");

  REQUIRE(root.has_group("a"));
  REQUIRE(a.has_group("b"));
  REQUIRE(b.has_group("c"));
  REQUIRE_FALSE(root.has_group("does_not_exist"));
  REQUIRE_FALSE(root.has_dataset("a"));

  Group b_again = a.open_group("b");
  REQUIRE(b_again.has_group("c"));

  REQUIRE_THROWS_AS(a.create_group("b"), std::runtime_error);
}

TEST_CASE("Datatype equality", "[hdf5lib][datatype]")
{
  CHECK(type_of<double>() == type_of<double>());
  CHECK(type_of<int>() == type_of<int>());
  CHECK_FALSE(type_of<double>() == type_of<float>());
  CHECK_FALSE(type_of<int>() == type_of<long>());

  // Compound equality is structural.
  auto build = []() {
    auto t = Datatype::compound(2 * sizeof(double));
    t.insert("re", 0, type_of<double>());
    t.insert("im", sizeof(double), type_of<double>());
    return t;
  };
  CHECK(build() == build());
}

TEST_CASE("Scalar I/O for builtins", "[hdf5lib][dataset]")
{
  ScratchFile sf("scalar_io");
  {
    File f = File::open(sf.str(), Access::Truncate);
    Group root = f.root();

    auto write_scalar = [&](const char *name, auto value) {
      auto type = type_of<decltype(value)>();
      auto space = Dataspace::scalar();
      auto ds = root.create_dataset(name, type, space);
      ds.write(value);
    };

    write_scalar("d", 3.14159);
    write_scalar("f", 2.71828f);
    write_scalar("i", -42);
    write_scalar("u", 99u);
    write_scalar("ll", static_cast<long long>(1ULL << 40));

    {
      auto type = type_of<std::string>();
      auto space = Dataspace::scalar();
      auto ds = root.create_dataset("s", type, space);
      ds.write(std::string("hello, λ世界"));
    }
  }
  {
    File f = File::open(sf.str(), Access::ReadOnly);
    Group root = f.root();

    double d;
    root.open_dataset("d").read(d);
    CHECK(d == Catch::Approx(3.14159));

    float fl;
    root.open_dataset("f").read(fl);
    CHECK(fl == Catch::Approx(2.71828f));

    int i;
    root.open_dataset("i").read(i);
    CHECK(i == -42);

    unsigned u;
    root.open_dataset("u").read(u);
    CHECK(u == 99u);

    long long ll;
    root.open_dataset("ll").read(ll);
    CHECK(ll == static_cast<long long>(1ULL << 40));

    std::string s;
    root.open_dataset("s").read(s);
    CHECK(s == "hello, λ世界");
  }
}

TEST_CASE("Vector I/O", "[hdf5lib][dataset]")
{
  ScratchFile sf("vector_io");

  auto round_trip_doubles = [&](const std::vector<double> &in, const char *name) {
    {
      File f = File::open(sf.str(), Access::Truncate);
      auto root = f.root();
      auto space = in.empty() ? Dataspace::simple({0}) : Dataspace::simple({in.size()});
      auto ds = root.create_dataset(name, type_of<double>(), space);
      ds.write(in);
    }
    File f = File::open(sf.str(), Access::ReadOnly);
    auto root = f.root();
    auto ds = root.open_dataset(name);
    std::vector<double> out(static_cast<std::size_t>(ds.dataspace().size()));
    if (!out.empty()) ds.read(out);
    REQUIRE(out == in);
  };

  round_trip_doubles({}, "empty");
  round_trip_doubles({42.0}, "one");
  std::vector<double> big(1000);
  for (std::size_t i = 0; i < big.size(); ++i) big[i] = std::sin(static_cast<double>(i));
  round_trip_doubles(big, "big");

  // vector<string>
  {
    std::vector<std::string> in{"alpha", "", "γ", "zeta"};
    {
      File f = File::open(sf.str(), Access::Truncate);
      auto root = f.root();
      auto ds = root.create_dataset("strs", type_of<std::string>(), Dataspace::simple({in.size()}));
      ds.write(in);
    }
    File f = File::open(sf.str(), Access::ReadOnly);
    auto root = f.root();
    auto ds = root.open_dataset("strs");
    std::vector<std::string> out(static_cast<std::size_t>(ds.dataspace().size()));
    ds.read(out);
    REQUIRE(out == in);
  }
}

TEST_CASE("Raw-pointer I/O", "[hdf5lib][dataset]")
{
  ScratchFile sf("ptr_io");
  std::vector<double> in(64);
  for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<double>(i) * 0.5;

  {
    File f = File::open(sf.str(), Access::Truncate);
    auto root = f.root();
    auto ds = root.create_dataset("buf", type_of<double>(), Dataspace::simple({in.size()}));
    ds.write(in.data(), in.size());
  }
  File f = File::open(sf.str(), Access::ReadOnly);
  std::vector<double> out(in.size());
  f.root().open_dataset("buf").read(out.data(), out.size());
  CHECK(out == in);
}

TEST_CASE("Chunked unlimited dataset (append pattern)", "[hdf5lib][dataset]")
{
  ScratchFile sf("chunked");
  constexpr std::size_t N = 300;
  {
    File f = File::open(sf.str(), Access::Truncate);
    auto root = f.root();
    auto space = Dataspace::simple_unlimited({1});
    auto ds = root.create_chunked_dataset("ts", type_of<double>(), space, {128});
    ds.write_at(0, 0.0);
    for (hsize_t i = 1; i < N; ++i) {
      ds.resize({i + 1});
      ds.write_at(i, static_cast<double>(i) * 0.25);
    }
  }
  File f = File::open(sf.str(), Access::ReadOnly);
  auto ds = f.root().open_dataset("ts");
  REQUIRE(static_cast<std::size_t>(ds.dataspace().size()) == N);
  std::vector<double> out(N);
  ds.read(out);
  for (std::size_t i = 0; i < N; ++i)
    CHECK(out[i] == Catch::Approx(static_cast<double>(i) * 0.25));
}

TEST_CASE("Soft link", "[hdf5lib][group]")
{
  ScratchFile sf("soft_link");
  {
    File f = File::open(sf.str(), Access::Truncate);
    auto root = f.root();
    auto a = root.create_group("a");
    auto x = a.create_dataset("x", type_of<int>(), Dataspace::scalar());
    x.write(7);

    auto b = root.create_group("b");
    b.create_soft_link("y", "/a/x");
  }
  File f = File::open(sf.str(), Access::ReadOnly);
  auto b = f.root().open_group("b");
  // The link target is reachable as if it were a child of b.
  auto y = b.open_dataset("y");
  int v;
  y.read(v);
  CHECK(v == 7);
}

TEST_CASE("Attributes round-trip", "[hdf5lib][attribute]")
{
  ScratchFile sf("attrs");
  {
    File f = File::open(sf.str(), Access::Truncate);
    auto root = f.root();
    auto g = root.create_group("g");
    g.write_attribute("d", 4.5);
    g.write_attribute("i", 17);
    g.write_attribute("s", std::string("hello"));
    g.write_attribute("cstr", "literal");
  }
  File f = File::open(sf.str(), Access::ReadOnly);
  auto g = f.root().open_group("g");
  CHECK(g.read_attribute<double>("d") == Catch::Approx(4.5));
  CHECK(g.read_attribute<int>("i") == 17);
  CHECK(g.read_attribute<std::string>("s") == "hello");
  CHECK(g.read_attribute<std::string>("cstr") == "literal");
  REQUIRE_THROWS_AS(g.read_attribute<int>("missing"), std::runtime_error);
}

TEST_CASE("Child iteration", "[hdf5lib][group]")
{
  ScratchFile sf("child_iter");
  {
    File f = File::open(sf.str(), Access::Truncate);
    auto root = f.root();
    root.create_group("g0");
    root.create_group("g1");
    root.create_group("g2");
    root.create_dataset("d0", type_of<double>(), Dataspace::scalar()).write(1.0);
    root.create_dataset("d1", type_of<int>(), Dataspace::scalar()).write(1);
  }
  File f = File::open(sf.str(), Access::ReadOnly);
  auto root = f.root();
  auto names = root.child_names();
  REQUIRE(names.size() == 5);
  std::sort(names.begin(), names.end());
  REQUIRE(names == std::vector<std::string>{"d0", "d1", "g0", "g1", "g2"});

  CHECK(root.child_is_group("g0"));
  CHECK(root.child_is_group("g1"));
  CHECK(root.child_is_dataset("d0"));
  CHECK_FALSE(root.child_is_group("d1"));
  CHECK_FALSE(root.child_is_dataset("g2"));
}

namespace
{
  struct Pair {
    double a;
    int b;
  };
} // namespace

template <> struct DiFfRG::hdf5::TypeTrait<Pair> {
  static Datatype get()
  {
    auto t = Datatype::compound(sizeof(Pair));
    t.insert("a", offsetof(Pair, a), type_of<double>());
    t.insert("b", offsetof(Pair, b), type_of<int>());
    return t;
  }
};

TEST_CASE("User-extensible TypeTrait (compound)", "[hdf5lib][datatype]")
{
  ScratchFile sf("user_compound");
  std::vector<Pair> in{{1.5, -1}, {2.5, 2}, {3.5, 3}};
  {
    File f = File::open(sf.str(), Access::Truncate);
    auto root = f.root();
    auto ds = root.create_dataset("p", type_of<Pair>(), Dataspace::simple({in.size()}));
    ds.write(in);
  }
  File f = File::open(sf.str(), Access::ReadOnly);
  auto ds = f.root().open_dataset("p");
  CHECK(ds.datatype() == type_of<Pair>());
  std::vector<Pair> out(in.size());
  ds.read(out);
  for (std::size_t i = 0; i < in.size(); ++i) {
    CHECK(out[i].a == Catch::Approx(in[i].a));
    CHECK(out[i].b == in[i].b);
  }
}

TEST_CASE("Error paths", "[hdf5lib][errors]")
{
  ScratchFile sf("errors");
  File f = File::open(sf.str(), Access::Truncate);
  auto root = f.root();

  REQUIRE_THROWS_AS(root.open_group("nope"), std::runtime_error);
  REQUIRE_THROWS_AS(root.open_dataset("nope"), std::runtime_error);

  auto ds = root.create_dataset("d", type_of<double>(), Dataspace::scalar());
  ds.write(1.0);
  CHECK(ds.datatype() == type_of<double>());
  CHECK_FALSE(ds.datatype() == type_of<int>());
}

TEST_CASE("Variable-length string reads do not leak", "[hdf5lib][string][slow]")
{
  ScratchFile sf("vlen_strings");
  constexpr std::size_t N = 2000;
  std::vector<std::string> in(N);
  for (std::size_t i = 0; i < N; ++i) in[i] = "row-" + std::to_string(i);

  {
    File f = File::open(sf.str(), Access::Truncate);
    auto root = f.root();
    auto ds = root.create_dataset("strs", type_of<std::string>(), Dataspace::simple({N}));
    ds.write(in);
  }

  for (int rep = 0; rep < 5; ++rep) {
    File f = File::open(sf.str(), Access::ReadOnly);
    auto ds = f.root().open_dataset("strs");
    std::vector<std::string> out(N);
    ds.read(out);
    REQUIRE(out.size() == N);
    REQUIRE(out.front() == in.front());
    REQUIRE(out.back() == in.back());
  }
}
