#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/models.hh>

#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/data/fe_output.hh>
#include <DiFfRG/discretization/data/hdf5_output.hh>
#include <DiFfRG/discretization/data/output_path.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/physics/interpolation.hh>

#include <algorithm>
#include <string>
#include <vector>

//--------------------------------------------------------------------------------------------
// A golden test for the .h5 tree DiFfRG writes.
//
// This exists to make "the on-disk layout is unchanged" a checkable claim rather than an
// aspiration, so that the HDF5 write path can be restructured (staging the writes, moving them
// onto a writer thread) without silently moving, renaming or retyping anything a downstream
// reader depends on -- HDF5Input, read.jl, and every plotting script outside this repository.
//
// It therefore asserts on the *structure*: which objects exist, where, of what kind, with what
// element type, shape and attributes, and that the soft links resolve. It deliberately does not
// assert on field values; the other tests in this directory cover those.
//
// When a layout change is genuinely intended, update the expectations here in the same commit
// and say so -- an edit to this file is the signal that readers need attention.
//--------------------------------------------------------------------------------------------

namespace
{
  using namespace DiFfRG;

  std::vector<std::string> sorted_children(DiFfRG::hdf5::Group &group)
  {
    // H5_ITER_NATIVE gives no ordering guarantee, so compare as a set.
    auto names = group.child_names();
    std::sort(names.begin(), names.end());
    return names;
  }

  /** Every immediate child, with its kind, so a group's shape can be asserted in one line. */
  void require_children(DiFfRG::hdf5::Group &group, const std::vector<std::string> &expected_groups,
                        const std::vector<std::string> &expected_datasets)
  {
    std::vector<std::string> expected;
    expected.insert(expected.end(), expected_groups.begin(), expected_groups.end());
    expected.insert(expected.end(), expected_datasets.begin(), expected_datasets.end());
    std::sort(expected.begin(), expected.end());

    CHECK(sorted_children(group) == expected);
    for (const auto &name : expected_groups)
      CHECK(group.child_is_group(name));
    for (const auto &name : expected_datasets)
      CHECK(group.child_is_dataset(name));
  }

  template <typename T>
  void require_dataset(DiFfRG::hdf5::Group &group, const std::string &name, const DiFfRG::hdf5::Dims &extents)
  {
    REQUIRE(group.has_dataset(name));
    auto dataset = group.open_dataset(name);
    CHECK(dataset.datatype() == DiFfRG::hdf5::type_of<T>());
    CHECK(dataset.dataspace().extents() == extents);
  }
} // namespace

TEST_CASE("HDF5 output layout is stable", "[output][hdf5][layout][golden]")
{
  DiFfRG::Init();

  using namespace dealii;
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using Discretization = CG::Discretization<Model, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;

  // 4 cells, fe_order 1, no subdivision: small enough that the node counts below are exact and
  // readable, which is the point of a golden test.
  ConfigTree config = json::value({{"physical", {}},
                                   {"discretization",
                                    {{"fe_order", 1},
                                     {"overintegration", 0},
                                     {"output_subdivisions", 1},
                                     {"EoM_abs_tol", 1e-10},
                                     {"EoM_max_iter", 0},
                                     {"grid", {{"x_grid", "0:0.25:1"}, {"y_grid", "0:1:1"}, {"z_grid", "0:1:1"}}}}},
                                   {"output", {{"verbosity", 0}, {"vtk", false}}}});

  auto output_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "DiFfRG_test_hdf5_layout");
  const auto &tmp = output_path.root();
  const std::string h5_name = "layout.h5";

  Testing::PhysicalParameters p_prm = {/*x0_initial = */ 0., /*x1_initial = */ 1.};
  Model model(p_prm);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(config)};
  Discretization discretization(mesh, config);
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  const VectorType &solution = initial_condition.spatial_data();

  constexpr std::size_t coord_size = 8;
  LogCoordinates log_coords(coord_size, 0.1, 10.0, 5.);
  using grid_point = device::array<typename LogCoordinates::ctype, LogCoordinates::dim>;
  const std::string coord_to_string = log_coords.to_string();
  SplineInterpolator1D<double, LogCoordinates> spline(log_coords);
  std::vector<double> spline_data(coord_size);
  for (std::size_t i = 0; i < coord_size; ++i)
    spline_data[i] = static_cast<double>(i * i);
  spline.update(spline_data.data());

  std::vector<double> direct_data(coord_size);
  for (std::size_t i = 0; i < coord_size; ++i)
    direct_data[i] = static_cast<double>(i) + 0.5;

  constexpr uint n_frames = 3;

  // Exercises every json::kind a configuration can hold, so the leaf mapping into /config is
  // pinned alongside the rest of the layout.
  const std::string configuration_json =
      R"({"golden":true,"a_double":1.5,"an_int":-7,"a_string":"text","an_array":[1,2,3],)"
      R"("nothing":null,"physical":{"Lambda":1e5,"nested":{"depth":2}}})";

  {
    HDF5Output hdf5_output(tmp.string(), h5_name, configuration_json);
    FEOutput<dim, VectorType> fe_output(tmp.string(), "layout", "fields", Config::OutputSettings(config));
    {
      auto root = hdf5_output.get_file().root();
      root.create_group("FE");
    }
    hdf5_output.close_file();
    fe_output.set_hdf5_output(&hdf5_output, /*session_closes_file = */ true);

    for (uint frame = 0; frame < n_frames; ++frame) {
      const double time = 0.5 * frame;

      hdf5_output.scalar("a_double", 1.5 * (frame + 1));
      hdf5_output.scalar("an_int", static_cast<int>(frame));
      hdf5_output.scalar("a_string", std::string("frame_") + std::to_string(frame));
      hdf5_output.scalar("a_complex", complex<double>(frame, -1. * frame));
      hdf5_output.scalar("an_array", std::array<double, 3>{{1., 2., 3.}});

      hdf5_output.map("spline", "log", spline);
      hdf5_output.map("direct", log_coords, direct_data.data());

      fe_output.attach(discretization.get_dof_handler(), solution, "u");
      fe_output.flush(time);

      hdf5_output.flush(time);
    }
    fe_output.finish();
  }

  // ---------------------------------------------------------------------------------------
  // Everything below is the expected tree.
  // ---------------------------------------------------------------------------------------
  auto file = DiFfRG::hdf5::File::open((tmp / h5_name).string(), DiFfRG::hdf5::Access::ReadOnly);
  auto root = file.root();

  SECTION("root")
  {
    require_children(root, {"FE", "config", "coordinates", "maps", "scalars"}, {});
    // The configuration lives in /config; the raw JSON string is deliberately not duplicated here.
    CHECK_FALSE(root.has_attribute("configuration_json"));

    // Written at creation and only set by mark_finished(), which nothing above calls -- this is
    // what a killed run leaves behind.
    CHECK(root.read_attribute<int>("finished") == 0);
    CHECK(root.read_attribute<int>("crashed") == 0);
  }

  SECTION("the configuration is mirrored as a browsable group tree")
  {
    auto config_group = root.open_group("config");
    require_children(config_group, {"physical"}, {});

    // HDF5 has no boolean type, so a bool lands as a 0/1 int.
    CHECK(config_group.read_attribute<int>("golden") == 1);
    CHECK(config_group.read_attribute<double>("a_double") == 1.5);
    CHECK(config_group.read_attribute<long long>("an_int") == -7);
    CHECK(config_group.read_attribute<std::string>("a_string") == "text");
    // Arrays are kept as their serialized JSON rather than becoming a second leaf kind.
    CHECK(config_group.read_attribute<std::string>("an_array") == "[1,2,3]");
    // A null carries no value, so no attribute is written for it.
    CHECK_FALSE(config_group.has_attribute("nothing"));

    auto physical = config_group.open_group("physical");
    require_children(physical, {"nested"}, {});
    CHECK(physical.read_attribute<double>("Lambda") == 1e5);
    CHECK(physical.open_group("nested").read_attribute<long long>("depth") == 2);
  }

  SECTION("scalars are one appendable dataset per name, one element per frame")
  {
    auto scalars = root.open_group("scalars");
    require_children(scalars, {}, {"a_complex", "a_double", "a_string", "an_array", "an_int", "time"});

    const DiFfRG::hdf5::Dims frames{n_frames};
    require_dataset<double>(scalars, "a_double", frames);
    require_dataset<int>(scalars, "an_int", frames);
    require_dataset<std::string>(scalars, "a_string", frames);
    require_dataset<complex<double>>(scalars, "a_complex", frames);
    require_dataset<std::array<double, 3>>(scalars, "an_array", frames);
    // "time" is appended by flush() itself, not by the caller.
    require_dataset<double>(scalars, "time", frames);

    std::vector<double> times(n_frames);
    scalars.open_dataset("time").read(times);
    CHECK(times == std::vector<double>{0., 0.5, 1.});
  }

  SECTION("coordinates are written once per distinct name")
  {
    auto coords = root.open_group("coordinates");

    // Two entries for one set of coordinates, because the two map() overloads name them
    // differently: map(name, coord_name, interpolator) uses the caller's coord_name, while
    // map(name, coordinates, data*) derives one from Coordinates::to_string(). That asymmetry
    // is the current on-disk contract, so it is pinned here rather than quietly tolerated.
    require_children(coords, {}, {coord_to_string, "log"});

    // A grid point is device::array<ctype, dim>, i.e. an HDF5 compound with one member per
    // dimension -- not a bare double, even in 1D.
    require_dataset<grid_point>(coords, "log", DiFfRG::hdf5::Dims{coord_size});
    require_dataset<grid_point>(coords, coord_to_string, DiFfRG::hdf5::Dims{coord_size});
  }

  SECTION("maps are one group per name, one subgroup per series")
  {
    auto maps = root.open_group("maps");
    require_children(maps, {"direct", "spline"}, {});

    for (const std::string &name : {"direct", "spline"}) {
      auto map_group = maps.open_group(name);
      require_children(map_group, {"0", "1", "2"}, {});

      for (uint frame = 0; frame < n_frames; ++frame) {
        auto series = map_group.open_group(std::to_string(frame));
        // "coordinates" is a soft link into /coordinates. Readers follow soft links
        // transparently -- HDF5Input and read.jl both just open_dataset("coordinates") -- which
        // is precisely why the same trick is available for the per-frame FE node array.
        require_children(series, {}, {"coordinates", "data", "series_numbers"});
        require_dataset<double>(series, "data", DiFfRG::hdf5::Dims{coord_size});
        require_dataset<grid_point>(series, "coordinates", DiFfRG::hdf5::Dims{coord_size});
        // One vlen string per series group, holding the series number as text.
        require_dataset<std::string>(series, "series_numbers", DiFfRG::hdf5::Dims{1});

        CHECK(series.read_attribute<double>("time") == 0.5 * frame);
        CHECK(series.read_attribute<int>("series_number") == static_cast<int>(frame));
      }
    }
  }

  SECTION("FE fields are one zero-padded group per frame")
  {
    auto fe = root.open_group("FE");
    require_children(fe, {"000000", "000001", "000002"}, {});

    // 4 cells, Q1, no subdivision, duplicate vertices deliberately not filtered out.
    constexpr hsize_t n_nodes = 8;

    for (uint frame = 0; frame < n_frames; ++frame) {
      auto series = fe.open_group(Utilities::int_to_string(frame, 6));
      require_children(series, {}, {"nodes", "u"});

      require_dataset<double>(series, "nodes", DiFfRG::hdf5::Dims{n_nodes, dim});
      require_dataset<double>(series, "u", DiFfRG::hdf5::Dims{n_nodes});

      CHECK(series.read_attribute<double>("time") == 0.5 * frame);
      CHECK(series.read_attribute<int>("series_number") == static_cast<int>(frame));
      CHECK(series.read_attribute<std::string>("output_name") == "layout");
    }
  }
}
