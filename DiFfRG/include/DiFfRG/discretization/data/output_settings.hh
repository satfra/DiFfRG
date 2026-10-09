#pragma once

#include <DiFfRG/common/config_tree.hh>

#include <spdlog/common.h>

#include <cstddef>
#include <string>

namespace DiFfRG
{
  namespace Config
  {
    /** Typed configuration consumed by output sinks. */
    struct OutputSettings {
      inline static constexpr std::size_t default_max_pending_bytes = 2ull * 1024ull * 1024ull * 1024ull;

      OutputSettings() = default;
      explicit OutputSettings(const ConfigTree &config);

      bool asynchronous = true;
      bool write_vtk = true;
      bool write_hdf5 = true;
      /** Mirrors /output/json. Forces the `<name>.log.json` copy of the configuration to be
       * written; without it the file is only produced when there is no HDF5 output to carry the
       * configuration in its `/config` group. */
      bool write_json = false;
      unsigned int max_pending_frames = 2;
      /** Frames the HDF5 writer thread may have in flight. Deliberately separate from
       * `max_pending_frames`, which bounds VTK/DataOut memory and has nothing to do with this.
       *
       * The default of 1 is a durability choice rather than a throughput one: each frame is
       * still closed on disk before the next is opened, so a hard abort (SIGKILL, node
       * eviction) loses at most this many frames on top of the one being written. Depth 1
       * already hides the whole open/write/close behind the next output interval, because
       * frames are separated by many timesteps. Ignored when `asynchronous` is false. */
      unsigned int hdf5_queue_depth = 1;
      std::size_t max_pending_bytes = default_max_pending_bytes;
      unsigned int subdivisions = 1;
      /** Mirrors /output/verbosity, which the timesteppers already read. It sets how much of the run log
       * reaches the console: at 0 only warnings and errors, at 1 the ordinary messages, from 2 upwards also
       * debug records. The log file always receives everything, independent of this.
       *
       * At 3 and above the output sinks additionally emit one per-frame timing line and one
       * output_timings.csv row. The aggregate timing report at the end of a run is a debug record:
       * it is always written to the log file, and reaches the console from verbosity 2 upwards. */
      int verbosity = 0;
      std::size_t log_queue_size = 8192;
      spdlog::level::level_enum log_level = spdlog::level::info;
      double Lambda = -1.;
      std::string configuration_json = "{}";
      std::string configuration_log = "{}";
    };
  } // namespace Config

  // Compatibility alias for applications written before typed settings were grouped under Config.
  using OutputSettings = Config::OutputSettings;
} // namespace DiFfRG
