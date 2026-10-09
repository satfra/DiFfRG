#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/discretization/data/hdf5_input.hh>
#include <DiFfRG/discretization/data/hdf5_output.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/physics/interpolation.hh>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test HDF5 input", "[input][hdf5]")
{
  DiFfRG::Init();

  using namespace DiFfRG;

  // Setup JSON configuration for HDF5Output
  ConfigTree json = json::value({{"physical", {}},
                                {"integration", {{"x_quadrature_order", 32}, {"angle_quadrature_order", 8}}},
                                {"discretization",
                                 {{"fe_order", 3},
                                  {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}}}}},
                                {"timestepping", {{"final_time", 1.}, {"output_dt", 1e-1}}},
                                {"output", {{"live_plot", false}, {"verbosity", 0}}}});

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
    log->info("DiFfRG HDF5Input test started");
  } catch (const spdlog::spdlog_ex &e) {
    // Logger already exists, continue
  }

  auto output_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "DiFfRG_test_hdf5_input");
  const auto &tmp = output_path.root();
  const std::string hdf5FileName = output_path.run_file(".h5").filename().string();
  const auto hdf5_file = output_path.run_file(".h5");

  SECTION("Test scalar loading")
  {
    // First, write test data using HDF5Output
    {
      HDF5Output hdf5_output(tmp.string(), hdf5FileName, Config::OutputSettings(json).configuration_json);

      // Write various scalar types (only basic types that work reliably)
      hdf5_output.scalar("test_double", 42.5);
      hdf5_output.scalar("test_int", 123);
      hdf5_output.scalar("test_string", "hello_world");

      hdf5_output.flush(1.0);
    } // HDF5Output destructor will close the file

    // Now read the data back using HDF5Input
    {
      HDF5Input hdf5_input(hdf5_file.string());

      // Test loading different scalar types
      auto double_vec = hdf5_input.load_scalar<double>("test_double");
      REQUIRE(double_vec.size() == 1);
      CHECK(double_vec[0] == Catch::Approx(42.5));

      auto int_vec = hdf5_input.load_scalar<int>("test_int");
      REQUIRE(int_vec.size() == 1);
      CHECK(int_vec[0] == 123);

      auto string_vec = hdf5_input.load_scalar<std::string>("test_string");
      REQUIRE(string_vec.size() == 1);
      CHECK(string_vec[0] == "hello_world");

      // Test error when scalar doesn't exist
      REQUIRE_THROWS_AS(hdf5_input.load_scalar<double>("nonexistent_scalar"), std::runtime_error);
    }
  }

  SECTION("Test map loading with coordinates")
  {
    // Create test coordinates
    LinearCoordinates1D<double> coords1(10, 0.0, 1.0);
    LinearCoordinates1D<double> coords2(5, -1.0, 1.0);
    CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>> coords2d(coords1, coords2);

    // Create test data
    std::vector<double> test_data(coords2d.size());
    for (size_t i = 0; i < coords2d.size(); ++i) {
      auto coord = coords2d.forward(coords2d.from_linear_index(i));
      test_data[i] = coord[0] * coord[0] + coord[1] * coord[1]; // x² + y²
    }

    // Write test data using HDF5Output
    {
      HDF5Output hdf5_output(tmp.string(), hdf5FileName, Config::OutputSettings(json).configuration_json);
      hdf5_output.map("test_map", coords2d, test_data.data());
      hdf5_output.flush(1.0);
    }

    // Read the data back using HDF5Input
    {
      HDF5Input hdf5_input(hdf5_file.string());
      std::vector<double> loaded_data(coords2d.size());

      // Test loading map with matching coordinates
      REQUIRE_NOTHROW(hdf5_input.load_map("test_map", loaded_data.data(), coords2d));

      // Verify the data matches
      for (size_t i = 0; i < coords2d.size(); ++i)
        CHECK(loaded_data[i] == Catch::Approx(test_data[i]));

      // Test loading map without coordinate checking
      std::vector<double> loaded_data2(coords2d.size());
      REQUIRE_NOTHROW(hdf5_input.load_map("test_map", loaded_data2.data()));

      // Verify the data matches
      for (size_t i = 0; i < coords2d.size(); ++i)
        CHECK(loaded_data2[i] == Catch::Approx(test_data[i]));

      // Test error when map doesn't exist
      std::vector<double> dummy_data(coords2d.size());
      REQUIRE_THROWS_AS(hdf5_input.load_map("nonexistent_map", dummy_data.data()), std::runtime_error);
    }
  }

  SECTION("Test map round trip on focused logarithmic coordinates")
  {
    // to_string() is the identity key of a coordinate dataset, so it has to separate grids that
    // differ only in the focusing parameters -- otherwise two different grids silently share one
    // dataset and load_map() accepts the wrong one.
    FocusedLogCoordinates1D<double> coords(32, 1e-3, 100.0, 1.0, 2.0);
    FocusedLogCoordinates1D<double> coords_other_center(32, 1e-3, 100.0, 5.0, 2.0);
    FocusedLogCoordinates1D<double> coords_other_focus(32, 1e-3, 100.0, 1.0, 3.0);

    CHECK(coords.to_string() != coords_other_center.to_string());
    CHECK(coords.to_string() != coords_other_focus.to_string());

    std::vector<double> test_data(coords.size());
    for (size_t i = 0; i < coords.size(); ++i)
      test_data[i] = std::log(coords.forward(i));

    {
      HDF5Output hdf5_output(tmp.string(), hdf5FileName, OutputSettings(json).configuration_json);
      hdf5_output.map("focused_map", coords, test_data.data());
      hdf5_output.flush(1.0);
    }

    {
      HDF5Input hdf5_input(hdf5_file.string());
      std::vector<double> loaded_data(coords.size());
      REQUIRE_NOTHROW(hdf5_input.load_map("focused_map", loaded_data.data(), coords));
      for (size_t i = 0; i < coords.size(); ++i)
        CHECK(loaded_data[i] == Catch::Approx(test_data[i]));

      // a grid with a different center or focus must be rejected
      std::vector<double> dummy_data(coords.size());
      REQUIRE_THROWS_AS(hdf5_input.load_map("focused_map", dummy_data.data(), coords_other_center),
                        std::runtime_error);
      REQUIRE_THROWS_AS(hdf5_input.load_map("focused_map", dummy_data.data(), coords_other_focus), std::runtime_error);
    }
  }

  SECTION("Test coordinate validation errors")
  {
    // Create original coordinates and different coordinates for testing validation
    LinearCoordinates1D<double> coords1(10, 0.0, 1.0);
    LinearCoordinates1D<double> coords2(5, -1.0, 1.0);
    CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>> coords2d(coords1, coords2);

    // Create different coordinates that should fail validation
    LinearCoordinates1D<double> wrong_coords1(8, 0.0, 1.0); // Different size
    LinearCoordinates1D<double> wrong_coords2(5, -1.0, 1.0);
    CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>> wrong_coords2d(wrong_coords1,
                                                                                              wrong_coords2);

    LinearCoordinates1D<double> wrong_range_coords1(10, 0.1, 1.0); // Different range
    LinearCoordinates1D<double> wrong_range_coords2(5, -1.0, 1.0);
    CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>> wrong_range_coords2d(
        wrong_range_coords1, wrong_range_coords2);

    // Create test data and write it
    std::vector<double> test_data(coords2d.size());
    for (size_t i = 0; i < coords2d.size(); ++i) {
      test_data[i] = static_cast<double>(i);
    }

    {
      HDF5Output hdf5_output(tmp.string(), hdf5FileName, Config::OutputSettings(json).configuration_json);
      hdf5_output.map("test_map_validation", coords2d, test_data.data());
      hdf5_output.flush(1.0);
    }

    // Test coordinate validation
    {
      HDF5Input hdf5_input(hdf5_file.string());
      std::vector<double> loaded_data(coords2d.size());

      // This should work with matching coordinates
      REQUIRE_NOTHROW(hdf5_input.load_map("test_map_validation", loaded_data.data(), coords2d));

      // This should fail with wrong size coordinates
      std::vector<double> wrong_size_data(wrong_coords2d.size());
      REQUIRE_THROWS_AS(hdf5_input.load_map("test_map_validation", wrong_size_data.data(), wrong_coords2d),
                        std::runtime_error);

      // This should fail with wrong range coordinates
      std::vector<double> wrong_range_data(wrong_range_coords2d.size());
      REQUIRE_THROWS_AS(hdf5_input.load_map("test_map_validation", wrong_range_data.data(), wrong_range_coords2d),
                        std::runtime_error);
    }
  }

  SECTION("Test missing coordinates error")
  {
    // Create coordinates that haven't been written to the file
    LinearCoordinates1D<double> coords1(6, 0.0, 2.0);
    LinearCoordinates1D<double> coords2(4, -2.0, 2.0);
    CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>> missing_coords(coords1, coords2);

    // Write some data with different coordinates
    LinearCoordinates1D<double> existing_coords1(5, 0.0, 1.0);
    std::vector<double> test_data(existing_coords1.size(), 1.0);

    {
      HDF5Output hdf5_output(tmp.string(), hdf5FileName, Config::OutputSettings(json).configuration_json);
      hdf5_output.map("test_map_missing_coords", existing_coords1, test_data.data());
      hdf5_output.flush(1.0);
    }

    // Try to read with coordinates that don't exist in the file
    {
      HDF5Input hdf5_input(hdf5_file.string());
      std::vector<double> loaded_data(missing_coords.size());

      // This should throw because the coordinates for missing_coords haven't been written
      REQUIRE_THROWS_AS(hdf5_input.load_map("test_map_missing_coords", loaded_data.data(), missing_coords),
                        std::runtime_error);
    }
  }

  SECTION("Test load_map_coord returns correct coordinate values")
  {
    LinearCoordinates1D<double> coords1(10, 0.0, 1.0);
    LinearCoordinates1D<double> coords2(5, -1.0, 1.0);
    CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>> coords2d(coords1, coords2);

    std::vector<double> test_data(coords2d.size(), 0.0);

    {
      HDF5Output hdf5_output(tmp.string(), hdf5FileName, Config::OutputSettings(json).configuration_json);
      hdf5_output.map("test_map_coord", coords2d, test_data.data());
      hdf5_output.flush(1.0);
    }

    {
      HDF5Input hdf5_input(hdf5_file.string());

      auto coord_vec = hdf5_input.load_map_coord<2>("test_map_coord");
      REQUIRE(coord_vec.size() == coords2d.size() * 2);

      // Verify spot values against make_grid
      const auto grid = make_grid(coords2d);
      for (size_t i = 0; i < coords2d.size(); ++i) {
        CHECK(coord_vec[2 * i + 0] == Catch::Approx(grid[i][0]));
        CHECK(coord_vec[2 * i + 1] == Catch::Approx(grid[i][1]));
      }
    }
  }
}
