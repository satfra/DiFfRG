#include <DiFfRG/discretization/data/output_settings.hh>

#include <sstream>

namespace DiFfRG::Config
{
  OutputSettings::OutputSettings(const ConfigTree &config)
      : write_vtk(config.get_bool("/output/vtk", true)), write_hdf5(config.get_bool("/output/hdf5", true)),
        write_json(config.get_bool("/output/json", false)),
        max_pending_frames(config.get_uint("/output/max_pending_frames", 2)),
        subdivisions(config.get_uint("/discretization/output_subdivisions", 1)),
        verbosity(config.get_int("/output/verbosity", 0)),
        log_queue_size(config.get_uint("/output/log_queue_size", 8192)),
        log_level(static_cast<spdlog::level::level_enum>(config.get_int("/output/log_level", SPDLOG_LEVEL_INFO))),
        Lambda(config.get_double("/physical/Lambda", -1.)),
        configuration_json(json::serialize(static_cast<json::value>(config)))
  {
    std::ostringstream stream;
    config.print(stream);
    configuration_log = stream.str();
  }
} // namespace DiFfRG::Config
