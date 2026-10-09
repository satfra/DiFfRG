#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/hdf5.hh>
#include <DiFfRG/physics/interpolation.hh>

#include <filesystem>
#include <list>

namespace DiFfRG
{
  /**
   * @brief A class to output data to a CSV file.
   *
   * In every time step, the user can add values to the output, which will be written to the file when flush is called.
   * Every timestep has to contain the same values, but the order in which they are added can change.
   */
  class HDF5Input
  {
  public:
    HDF5Input(const std::string file_name);

    /**
     * @brief Load a map from the HDF5 file.
     *
     * @param name The name of the map.
     * @param data The data buffer to load the map into.
     */
    void load_map(const std::string &name, double *data, int series_number = -1)
    {
      if (!maps.has_group(name))
        throw std::runtime_error("HDF5Input::map: The map '" + name + "' has not been written to the file '" +
                                 file_name + "'.");

      auto super_group = maps.open_group(name);
      if (series_number < 0) {
        size_t max_series = 0;
        for (const auto &child : super_group.child_names()) {
          if (!super_group.child_is_group(child)) continue;
          size_t grp_num = std::stoul(child);
          if (grp_num > max_series) {
            max_series = grp_num;
          }
        }
        series_number = static_cast<int>(max_series);
      }
      if (!super_group.has_group(std::to_string(series_number)))
        throw std::runtime_error("HDF5Input::map: The map '" + name + "' does not have series number " +
                                 std::to_string(series_number) + " in the file '" + file_name + "'.");

      auto group = super_group.open_group(std::to_string(series_number));

      auto dataset = group.open_dataset("data");
      auto dataspace = dataset.dataspace();
      std::vector<double> temp_data(dataspace.size());
      dataset.read(temp_data);
      std::copy(temp_data.begin(), temp_data.end(), data);
    }

    /**
     * @brief Load a map from the HDF5 file.
     *
     * @param name The name of the map.
     * @param data The data buffer to load the map into.
     */
    std::vector<double> load_map(const std::string &name, int series_number = -1)
    {
      if (!maps.has_group(name))
        throw std::runtime_error("HDF5Input::map: The map '" + name + "' has not been written to the file '" +
                                 file_name + "'.");

      auto super_group = maps.open_group(name);
      if (series_number < 0) {
        size_t max_series = 0;
        for (const auto &child : super_group.child_names()) {
          if (!super_group.child_is_group(child)) continue;
          size_t grp_num = std::stoul(child);
          if (grp_num > max_series) {
            max_series = grp_num;
          }
        }
        series_number = static_cast<int>(max_series);
      }
      if (!super_group.has_group(std::to_string(series_number)))
        throw std::runtime_error("HDF5Input::map: The map '" + name + "' does not have series number " +
                                 std::to_string(series_number) + " in the file '" + file_name + "'.");

      auto group = super_group.open_group(std::to_string(series_number));

      auto dataset = group.open_dataset("data");
      auto dataspace = dataset.dataspace();
      std::vector<double> temp_data(dataspace.size());
      dataset.read(temp_data);
      return temp_data;
    }

    /**
     * @brief Load a map's coordinates from the HDF5 file.
     *
     * @tparam dim The spatial dimension of the coordinate entries.
     * @param name The name of the map.
     * @param series_number The series number to load (-1 for latest).
     * @return A flattened vector of doubles with size = num_points * dim.
     */
    template <size_t dim>
    std::vector<double> load_map_coord(const std::string &name, int series_number = -1)
    {
      if (!maps.has_group(name))
        throw std::runtime_error("HDF5Input::map: The map '" + name + "' has not been written to the file '" +
                                 file_name + "'.");

      auto super_group = maps.open_group(name);
      if (series_number < 0) {
        size_t max_series = 0;
        for (const auto &child : super_group.child_names()) {
          if (!super_group.child_is_group(child)) continue;
          size_t grp_num = std::stoul(child);
          if (grp_num > max_series) {
            max_series = grp_num;
          }
        }
        series_number = static_cast<int>(max_series);
      }
      if (!super_group.has_group(std::to_string(series_number)))
        throw std::runtime_error("HDF5Input::map: The map '" + name + "' does not have series number " +
                                 std::to_string(series_number) + " in the file '" + file_name + "'.");

      auto group = super_group.open_group(std::to_string(series_number));

      auto dataset = group.open_dataset("coordinates");
      auto dataspace = dataset.dataspace();

      using coord_type = device::array<double, dim>;
      std::vector<coord_type> compound_data(dataspace.size());
      dataset.read(compound_data);

      std::vector<double> result;
      result.reserve(dataspace.size() * dim);
      for (const auto &pt : compound_data)
        for (size_t d = 0; d < dim; ++d)
          result.push_back(pt[d]);
      return result;
    }

    /**
     * @brief Load a map from the HDF5 file, while checking that the coordinates in the file match the provided
     * coordinates.
     *
     * @param name The name of the map.
     * @param data The data buffer to load the map into.
     * @param coordinates The coordinates of the map.
     */
    template <typename Coordinates> void load_map(const std::string &name, double *data, const Coordinates &coordinates)
    {
      check_coordinates(coordinates.to_string(), coordinates);
      load_map(name, data);
    }

    template <typename T> std::vector<T> load_scalar(const std::string &name)
    {
      if (!scalars.has_dataset(name))
        throw std::runtime_error("HDF5Input::scalar: The scalar '" + name + "' has not been written to the file '" +
                                 file_name + "'.");

      auto dataset = scalars.open_dataset(name);
      auto dataspace = dataset.dataspace();
      std::vector<T> value(dataspace.size());
      dataset.read(value);
      return value;
    }

    DiFfRG::hdf5::File &get_file();

  private:
    const std::string file_name;

    template <typename Coordinates>
    void check_coordinates(const std::string &coord_name, const Coordinates &coordinates)
    {
      if (!coords.has_dataset(coord_name))
        throw std::runtime_error("HDF5Input::map: The coordinates '" + coord_name +
                                 "' have not been written to the file '" + file_name + "'.");

      const auto in_data = make_grid(coordinates);

      using ctype = typename Coordinates::ctype;
      using cortype = device::array<ctype, Coordinates::dim>;
      std::vector<cortype> file_data(coordinates.size());

      auto dataset = coords.open_dataset(coord_name);
      const auto dataspace = dataset.dataspace();
      if (dataspace.size() != (hssize_t)coordinates.size())
        throw std::runtime_error("HDF5Input::map: The coordinates '" + coord_name + "' in the file '" + file_name +
                                 "' have incorrect size (" + std::to_string(dataspace.size()) +
                                 "), not matching the provided coordinates (" + std::to_string(coordinates.size()) +
                                 ").");
      dataset.read(file_data);

      for (size_t i = 0; i < coordinates.size(); ++i) {
        for (size_t j = 0; j < Coordinates::dim; ++j) {
          if (std::abs(file_data[i][j] - in_data[i][j]) > 1e-12) {
            throw std::runtime_error("HDF5Input::map: The coordinates '" + coord_name +
                                     "' do not match the coordinates in the file '" + file_name + "'.");
          }
        }
      }
    }

    DiFfRG::hdf5::File h5_file;
    DiFfRG::hdf5::Group scalars;
    DiFfRG::hdf5::Group maps;
    DiFfRG::hdf5::Group coords;

    std::filesystem::path path;
  };
} // namespace DiFfRG
