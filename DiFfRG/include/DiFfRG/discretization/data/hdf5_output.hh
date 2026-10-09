#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/hdf5.hh>
#include <DiFfRG/discretization/data/hdf5_frame.hh>
#include <DiFfRG/discretization/data/hdf5_writer.hh>
#include <DiFfRG/discretization/data/output_settings.hh>
#include <DiFfRG/discretization/data/output_timings.hh>
#include <DiFfRG/physics/interpolation.hh>

#include <filesystem>
#include <list>
#include <set>
#include <typeindex>
#include <utility>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief Mirror a JSON configuration value into `group` as a browsable tree: one subgroup per
   * object, one attribute per leaf.
   *
   * This is how a run's configuration is recorded in its HDF5 file, so that it reads with the same
   * tools as its data. HDF5 has no boolean type, so booleans become 0/1 ints, arrays are stored as
   * their serialized JSON, and nulls are skipped; set /output/json to also get the verbatim
   * `<name>.log.json` copy alongside.
   *
   * @throws std::runtime_error if an object key contains '/', which is not a legal HDF5 link name.
   */
  void write_config_tree(DiFfRG::hdf5::Group &group, const json::value &value);

  /**
   * @brief A class to output data to a CSV file.
   *
   * In every time step, the user can add values to the output, which will be written to the file when flush is called.
   * Every timestep has to contain the same values, but the order in which they are added can change.
   */
  class HDF5Output
  {
  public:
    /**
     * @brief Construct a new Csv Output object
     *
     * @param top_folder The top folder to store the output in.
     * @param output_name The name of the output file.
     * @param config The ConfigTree object containing the parameters.
     */
    HDF5Output(const std::string top_folder, const std::string output_name,
               const std::string &configuration_json = "{}");

    [[deprecated("Use HDF5Output(top_folder, output_name, Config::OutputSettings(config).configuration_json) instead")]]
    HDF5Output(const std::string top_folder, const std::string output_name, const ConfigTree &config)
        : HDF5Output(std::move(top_folder), std::move(output_name), Config::OutputSettings(config).configuration_json)
    {
    }

    ~HDF5Output();

    /** Append the series number, as text, to a per-series-group dataset. Runs on the writer. */
    static void write_series_record(DiFfRG::hdf5::Group &group, const int series_number)
    {
      const std::string value = std::to_string(series_number);

      if (!group.has_dataset("series_numbers")) {
        // chunked layout so we can append later
        auto space = DiFfRG::hdf5::Dataspace::simple_unlimited({1});
        auto type = DiFfRG::hdf5::type_of<std::string>();
        auto data_set = group.create_chunked_dataset("series_numbers", type, space, {128});
        data_set.write_at(0, value);
      } else {
        auto data_set = group.open_dataset("series_numbers");
        const std::size_t cur_size = static_cast<std::size_t>(data_set.dataspace().size());
        data_set.resize({cur_size + 1});
        data_set.write_at(static_cast<hsize_t>(cur_size), value);
      }
    }

    /**
     * @brief Add a value to the output.
     *
     * Validates and copies here, on the calling thread, so that a rejected value still fails at
     * the call site with the context that produced it -- but performs no HDF5 call. The write
     * itself is staged and happens when the frame is flushed.
     *
     * @param name The name of the value.
     * @param value The value to add.
     */
    template <typename T> void scalar(const std::string &name, const T value)
    {
      using value_type = std::decay_t<T>;

      if (initial_scalars.size() > 0 &&
          std::find(initial_scalars.begin(), initial_scalars.end(), name) == initial_scalars.end())
        throw std::runtime_error("HDF5Output::scalar: The scalar '" + name + "' has not been registered before!");

      // Every check happens before any bookkeeping is touched: a rejected write must leave the
      // frame exactly as it was, so that a later, valid write of the same name still succeeds.
      auto info = scalar_infos.find(name);
      if (info != scalar_infos.end()) {
        // std::type_index rather than H5Tequal, so that no HDF5 call is needed to validate.
        // It is strictly finer than H5Tequal: two distinct C++ types mapping onto the same HDF5
        // type would now be rejected. No such pair exists among the traits in hdf5.hh.
        if (info->second.type != std::type_index(typeid(value_type)))
          throw std::runtime_error(
              "HDF5Output::scalar: The type of the value does not match the type of the dataset '" + name +
              "' in the file '" + output_name + "'.");

        if (std::find(written_scalars.begin(), written_scalars.end(), name) != written_scalars.end())
          throw std::runtime_error("HDF5Output::scalar: The scalar '" + name +
                                   "' has already been written to the file '" + output_name + "'.");
      } else {
        info = scalar_infos.emplace(name, ScalarInfo{std::type_index(typeid(value_type)), 0}).first;
      }

      written_scalars.push_back(name);
      const bool create = info->second.count == 0;
      const hsize_t offset = static_cast<hsize_t>(info->second.count++);

      staged.actions.emplace_back([name, v = value_type(value), create, offset](HDF5FrameContext &context) {
        auto &group = context.group("scalars");
        auto type = DiFfRG::hdf5::type_of<value_type>();
        if (create) {
          auto space = DiFfRG::hdf5::Dataspace::simple_unlimited({1});
          auto data_set = group.create_chunked_dataset(name, type, space, {128});
          data_set.write_at(0, v);
        } else {
          auto data_set = group.open_dataset(name);
          data_set.resize({offset + 1});
          data_set.write_at(offset, v);
        }
      });
    }
    void scalar(const std::string &name, const char *value) { scalar<std::string>(name, std::string(value)); }

    template <typename COORD>
      requires is_coordinates<COORD>
    void map(const std::string &name, const COORD &coordinates)
    {
      // In-memory, because staging must not touch the file at all. It also fixes the original:
      // this asked has_dataset() on a group handle that was closed between frames, which
      // silently answered "no" and turned the diagnostic into a raw H5Dcreate2 failure.
      if (coord_identifiers.find(name) != coord_identifiers.end()) {
        throw std::runtime_error("HDF5Output::map: The coordinates '" + name +
                                 "' have already been written to the file '" + output_name + "'.");
      }

      // Materialised here because it reads `coordinates`, which the caller owns; the write of
      // the resulting buffer is staged.
      auto grid_data = make_grid(coordinates);
      using coord_type = typename decltype(grid_data)::value_type;

      DiFfRG::hdf5::Dims dims;
      for (const auto &dim : coordinates.sizes())
        dims.push_back(dim);

      coord_identifiers[name] = coordinates.to_string();

      staged.actions.emplace_back([name, dims, grid = std::move(grid_data)](HDF5FrameContext &context) {
        auto space = DiFfRG::hdf5::Dataspace::simple(dims);
        auto dataset = context.group("coordinates").create_dataset(name, DiFfRG::hdf5::type_of<coord_type>(), space);
        dataset.write(grid);
      });
    }

    template <typename COORD, typename T>
      requires is_coordinates<COORD>
    void map(const std::string &name, const COORD &coordinates, T *data)
    {
      // Note the asymmetry with the interpolator overload below: that one takes the coordinate
      // dataset's name from the caller, this one derives it from the coordinates themselves.
      // Both spellings therefore appear under /coordinates. Preserved deliberately -- the
      // golden layout test pins it.
      const std::string coord_name = coordinates.to_string();
      if (coord_identifiers.find(coord_name) == coord_identifiers.end()) map(coord_name, coordinates);

      using value_type = std::decay_t<T>;

      DiFfRG::hdf5::Dims dims;
      for (const auto &dim : coordinates.sizes())
        dims.push_back(dim);

      // `data` points into live simulation state, so the copy is mandatory, not defensive.
      std::vector<value_type> payload(data, data + coordinates.size());
      stage_map(name, coord_name, dims, std::move(payload));
    }

    template <typename INTERP>
      requires is_interpolator<INTERP>
    void map(const std::string &name, const INTERP &_interpolator)
    {
      map(name, _interpolator.get_coordinates().to_string(), _interpolator);
    }

    template <typename INTERP>
      requires is_interpolator<INTERP>
    void map(const std::string &name, const std::string &coord_name, const INTERP &_interpolator)
    {
      using value_type = typename std::decay_t<INTERP>::value_type;

      const auto &interpolator = _interpolator;

      check_coordinates(coord_name, interpolator.get_coordinates());

      DiFfRG::hdf5::Dims dims;
      const auto coordinates = interpolator.get_coordinates();
      for (const auto &dim : coordinates.sizes())
        dims.push_back(dim);

      // Sampling stays here rather than moving into the staged action; only the resulting dense
      // buffer is deferred.
      std::vector<value_type> payload(coordinates.size());
      for (size_t i = 0; i < payload.size(); ++i) {
        const auto lcoord = coordinates.forward(coordinates.from_linear_index(i));
        payload[i] = device::apply([&](const auto &...x) { return interpolator(x...); }, lcoord);
      }

      stage_map(name, coord_name, dims, std::move(payload));
    }

    /**
     * @brief Stage the node coordinates and nodal values of one finite-element frame.
     *
     * Called by FEOutput, which owns the DataOut and the filtering, but not the file. The
     * arrays are moved in; nothing here touches HDF5.
     */
    void stage_fe_frame(unsigned int series_number, double time, const std::string &group_name,
                        const std::string &fe_output_name, unsigned int dim, std::size_t n_nodes,
                        std::vector<double> node_data,
                        std::vector<std::pair<std::string, std::vector<double>>> data_sets);

    /**
     * @brief Open the file for setup work performed before any frame is written.
     *
     * Not for per-frame writes -- those go through scalar()/map()/stage_fe_frame() and are
     * executed together when the frame is flushed. Callers must close_file() when done, so the
     * lock is not held into the run.
     */
    DiFfRG::hdf5::File &get_file();

    void flush(const double time);

    /**
     * @brief Execute and clear whatever is staged, without the per-frame scalar bookkeeping.
     *
     * flush() is the normal entry point. This exists for the standalone FEOutput case, where
     * nobody flushes the HDF5Output at all and the staged field data would otherwise be
     * discarded at destruction.
     */
    void run_staged_frame(double time);

    /**
     * @brief Route staged frames through a writer thread instead of writing them inline.
     *
     * Not owned. Every HDF5Output of one run must share the same writer, because HDF5 is not
     * thread safe here and two writers would call into it concurrently. Passing nullptr (the
     * default) keeps everything on the calling thread.
     */
    void set_writer(HDF5FrameWriter *writer) { this->writer = writer; }

    /** Wait for this sink's queued frames to reach disk and surface any writer error. */
    void drain();

    /**
     * @brief Record how the run ended: sets the root `finished` attribute to 1, and `crashed` to
     * whether it ended by throwing.
     *
     * Call once, after the last frame. Files start at `finished = 0`, so a run that never gets
     * here -- SIGKILL, OOM, a node eviction -- stays marked unfinished. A run that threw still
     * reaches this and is marked finished, with `crashed = 1`: its output was flushed and closed
     * properly, it just stopped early.
     */
    void mark_finished(bool crashed);

    void open_file();
    void close_file();

    /** Timing buckets accumulated since the last call, and reset by it. */
    FrameTimings take_frame_timings()
    {
      const FrameTimings taken = frame_timings;
      frame_timings = FrameTimings{};
      return taken;
    }

  private:
    /**
     * @brief Stage one map series: the group, its series record, its data and its coordinate link.
     *
     * Shared by the raw-pointer and interpolator overloads, which differ only in how they
     * produce the dense payload and how the coordinate dataset gets its name.
     */
    template <typename T>
    void stage_map(const std::string &name, const std::string &coord_name, const DiFfRG::hdf5::Dims &dims,
                   std::vector<T> payload)
    {
      const std::size_t series = map_series_numbers[name]++;
      // Whether this frame has to create the per-name parent group is decided here rather than
      // asked of the file, so that staging stays HDF5-free.
      const bool create_parent = created_map_groups.insert(name).second;

      written_map_series.emplace_back(name, series);
      staged.actions.emplace_back(
          [name, coord_name, dims, series, create_parent, data = std::move(payload)](HDF5FrameContext &context) {
            auto &maps_group = context.group("maps");
            auto parent = create_parent ? maps_group.create_group(name) : maps_group.open_group(name);
            auto group = parent.create_group(std::to_string(series));
            write_series_record(group, static_cast<int>(series));

            auto space = DiFfRG::hdf5::Dataspace::simple(dims);
            auto dataset = group.create_dataset("data", DiFfRG::hdf5::type_of<T>(), space);
            dataset.write(data);

            group.create_soft_link("coordinates", "/coordinates/" + coord_name);
          });
    }

    const std::string top_folder;
    const std::string output_name;

    bool opened;

    /** Element type and how many elements have been staged, per scalar name. Replaces asking
     * the file whether a dataset exists and what type and length it has. */
    struct ScalarInfo {
      std::type_index type;
      std::size_t count = 0;
    };
    std::map<std::string, ScalarInfo> scalar_infos;
    /** Map names whose parent group has been created (or staged for creation). */
    std::set<std::string> created_map_groups;

    /** Writes accumulated for the frame in progress; executed by flush(). */
    HDF5Frame staged;
    /** Not owned; shared with every other HDF5Output of the same run. */
    HDF5FrameWriter *writer = nullptr;

    std::list<std::string> written_scalars;
    /** (name, series) of every map written this frame. Carries the series number rather than
     * re-deriving it as `map_series_numbers[name] - 1` in flush(): a name written twice in one
     * frame would otherwise stamp attributes onto the same group twice and fail in H5Acreate2,
     * pointing at the flush instead of at the duplicate write. */
    std::vector<std::pair<std::string, size_t>> written_map_series;
    std::map<std::string, size_t> map_series_numbers;
    std::map<std::string, std::string> coord_identifiers;

    std::list<std::string> initial_scalars;

    template <typename COORD> void check_coordinates(const std::string &coord_name, const COORD &coordinates)
    {
      // In-memory bookkeeping rather than a file query, for the reason given in map().
      if (coord_identifiers.find(coord_name) == coord_identifiers.end()) {
        map(coord_name, coordinates);
        return;
      }

      if (coordinates.to_string() != coord_identifiers[coord_name])
        throw std::runtime_error("HDF5Output::map: The coordinates '" + coord_name +
                                 "' do not match the interpolator's coordinates.");
    }

    /** Only ever open for setup work (see get_file()); frames use their own context. */
    DiFfRG::hdf5::File h5_file;

    std::filesystem::path path;

    FrameTimings frame_timings;
  };
} // namespace DiFfRG
