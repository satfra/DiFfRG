#pragma once

#include <DiFfRG/common/config_tree.hh>

#include <spdlog/common.h>

#include <cstddef>
#include <string>

namespace DiFfRG
{
  namespace Config
  {
    // clang-format off
    /**
     * @brief Settings of the output sinks (OutputSession, FEOutput, HDF5Output).
     *
     * | Key                                 | Type   | Default | Member / meaning |
     * |-------------------------------------|--------|---------|------------------|
     * | `/output/vtk`                       | bool   | true    | write_vtk: VTK files of the FE fields |
     * | `/output/hdf5`                      | bool   | true    | write_hdf5: the run's `.h5` file |
     * | `/output/json`                      | bool   | false   | write_json: write `<run>.log.json` (always written if write_hdf5 is false) |
     * | `/output/max_pending_frames`        | uint   | 2       | VTK frames written in the background at once (at least 2 are used) |
     * | `/output/verbosity`                 | int    | 0       | console log: 0 warnings and errors, 1 messages, ≥2 debug; ≥3 also per-frame output timings |
     * | `/output/log_queue_size`            | uint   | 8192    | message queue length of the asynchronous logger (at least 256) |
     * | `/output/log_level`                 | int    | 2       | spdlog level of the log file `<run>.log` (0 trace, 1 debug, 2 info, ...) |
     * | `/discretization/output_subdivisions` | uint | 1       | subdivisions: cells are subdivided this often in the FE output |
     * | `/physical/Lambda`                  | double | -1      | Lambda: if > 0, CSV tables get a column `k [GeV]` |
     *
     * `asynchronous`, `hdf5_queue_depth` and `max_pending_bytes` are not read from the parameters; set them in code.
     */
    // clang-format on
    struct OutputSettings {
      /// Default of max_pending_bytes: 2 GiB.
      inline static constexpr std::size_t default_max_pending_bytes = 2ull * 1024ull * 1024ull * 1024ull;

      /// All defaults.
      OutputSettings() = default;
      /// Read the keys of the table above.
      explicit OutputSettings(const ConfigTree &config);

      /// Write VTK and HDF5 frames on background threads (false: on the calling thread).
      bool asynchronous = true;
      bool write_vtk = true;
      bool write_hdf5 = true;
      bool write_json = false;
      unsigned int max_pending_frames = 2;
      /// HDF5 frames queued for the writer thread at most; ignored if asynchronous is false.
      unsigned int hdf5_queue_depth = 1;
      /// Memory bound of the field copies held by pending VTK frames (0: no bound).
      std::size_t max_pending_bytes = default_max_pending_bytes;
      unsigned int subdivisions = 1;
      /// At ≥ 3, each frame's output timings are logged and appended to `<run>_output_timings.csv`.
      int verbosity = 0;
      std::size_t log_queue_size = 8192;
      spdlog::level::level_enum log_level = spdlog::level::info;
      double Lambda = -1.;
      /// The full parameters as JSON (written to `/config` of the HDF5 files).
      std::string configuration_json = "{}";
      /// The full parameters as printed text (written to `<run>.log.json`).
      std::string configuration_log = "{}";
    };
  } // namespace Config

  /// Alias of Config::OutputSettings.
  using OutputSettings = Config::OutputSettings;
} // namespace DiFfRG
