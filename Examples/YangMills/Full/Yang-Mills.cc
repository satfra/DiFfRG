#include <filesystem>

#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/csv_reader.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/timestepping/timestepping.hh>

#include "model.hh"
#include "tuning.hh"

using namespace dealii;
using namespace DiFfRG;

// Choices for types
using Model = YangMills;
using VectorType = Vector<double>;
using Assembler = Variables::Assembler<Model>;
using TimeStepper = TimeStepperBoostABM<VectorType>;

bool run(const JSONValue &json, const std::string logger)
{
  const ConfigurationHelper config_helper(json);

  // Define the objects needed to run the simulation
  Model model(json);
  Assembler assembler(model, json);
  TimeStepper time_stepper(json, &assembler);

  // Set up the initial condition
  FlowingVariables initial_condition;
  initial_condition.interpolate(model);

  // Start the timestepping
  try {
    time_stepper.run(&initial_condition, 0., json.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    spdlog::get(logger)->error("Timestepping finished with exception {}", e.what());
    spdlog::get(logger)->flush();
    return false;
  }

  {
    CSVReader reader(config_helper.get_top_folder() + "/strong_couplings.csv", ',', true);
    uint first_p_idx = reader.n_rows() - 1;
    double min_p = std::numeric_limits<double>::max();
    for (uint i = reader.n_rows() - 1; i > 0; --i) {
      if (reader.value("pGeV", i) < min_p) {
        min_p = reader.value("pGeV", i);
        first_p_idx = i;
      } else
        break;
    }
    const double first_alphaAcbc = reader.value("alphaAcbc", first_p_idx);
    const double second_alphaAcbc = reader.value("alphaAcbc", first_p_idx + 3);
    const double third_alphaAcbc = reader.value("alphaAcbc", first_p_idx + 6);
    if (first_alphaAcbc > second_alphaAcbc || second_alphaAcbc > third_alphaAcbc) {
      spdlog::get(logger)->error("Timestepping finished beyond scaling.");
      spdlog::get(logger)->flush();
      return false;
    }

    CSVReader reader_Zs(config_helper.get_top_folder() + "/Zs.csv", ',', true);
    const auto Zc_out = reader_Zs.value("Zc", first_p_idx);
    if (Zc_out < 0) {
      spdlog::get(logger)->error("Timestepping finished with negative Zc.");
      spdlog::get(logger)->flush();
      return false;
    }
  }

  {
    CSVReader reader(config_helper.get_top_folder() + config_helper.get_output_name() + "_data.csv", ',', true);
    const auto m2A_out = reader.value("m2A", reader.n_rows() - 1);
    if (std::abs(m2A_out) > powr<2>(json.get_double("/physical/Lambda") / 2.)) {
      spdlog::get(logger)->error("Timestepping finished with jumping m2A.");
      spdlog::get(logger)->flush();
      return false;
    }
  }

  spdlog::get(logger)->error("Timestepping finished successfully.");
  return true;
}

int main(int argc, char *argv[])
{
  Timer timer;

  // get all needed parameters and parse from the CLI
  ConfigurationHelper config_helper(argc, argv);
  auto json = config_helper.get_json();

  if (json.get_bool("/tuning/tune_STI")) {
    tune_STI(json, config_helper.get_top_folder(), config_helper.get_output_name(), run);
  } else if (json.get_bool("/tuning/tune_m2A")) {
    tune_m2A(json, config_helper.get_top_folder(), config_helper.get_output_name(), run);
  } else {
    const auto ret_val = run(json, "log");
    // We print a bit of exit information.
    const auto time = timer.wall_time();
    spdlog::get("log")->info("Program finished after " + time_format(time));
    return ret_val;
  }

  // We print a bit of exit information.
  const auto time = timer.wall_time();
  spdlog::get("log")->info("Program finished after " + time_format(time));
  return 0;
}
