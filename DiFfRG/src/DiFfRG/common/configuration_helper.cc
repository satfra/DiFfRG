// external libraries
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  bool ConfigurationHelper::logger_initialized = false;

  ConfigurationHelper::ConfigurationHelper(int argc, char *argv[], const std::string parameter_file)
      : parsed(false), parameter_file(parameter_file)
  {
    for (int i = 1; i < argc; ++i)
      args.emplace_back(argv[i]);
    try {
      parse();
    } catch (const std::exception &e) {
      std::cerr << "While reading the CLI arguments an error occured:\n    " << e.what() << "\n" << std::endl;
      print_usage_message();
      exit(0);
    }

    create_folder(get_top_folder());
    setup_logging();
  }

  ConfigurationHelper::ConfigurationHelper(const JSONValue &json) : json(json), parsed(true)
  {
    create_folder(get_top_folder());
    setup_logging();
  }

  void ConfigurationHelper::setup_logging()
  {
    if (!parsed) throw std::runtime_error("The ConfigurationHelper has to be parsed before the log can be created!");

    if (!logger_initialized) {
      spdlog::flush_every(std::chrono::seconds(1));

      try {
        auto console = spdlog::stdout_color_mt("console");
        console->set_pattern("[%v]");
      } catch (const spdlog::spdlog_ex &e) {
        // nothing, the logger is already set up
      }

      try {
        build_logger("log", get_top_folder() + get_log_file());
      } catch (const spdlog::spdlog_ex &e) {
        // nothing, the logger is already set up
      }

      logger_initialized = true;
    }

    auto jsonlog_filestream = std::ofstream(get_top_folder() + get_output_name() + ".log.json");
    json.print(jsonlog_filestream);
  }

  ConfigurationHelper::ConfigurationHelper(const ConfigurationHelper &other)
  {
    if (other.parsed) throw std::runtime_error("A parsed ConfigurationHelper can not be copied!");

    parameter_file = other.parameter_file;

    args = other.args;
    cli_parameters = other.cli_parameters;

    depth_console = other.depth_console;
    depth_file = other.depth_file;
  }

  JSONValue &ConfigurationHelper::get_json() { return json; }
  const JSONValue &ConfigurationHelper::get_json() const { return json; }

  void ConfigurationHelper::parse()
  {
    parsed = true;

    try {
      parse_cli();
    } catch (const std::exception &e) {
      std::cerr << "While reading the CLI arguments an error occured:\n    " << e.what() << "\n" << std::endl;
      print_usage_message();
      exit(0);
    }

    try {
      if (parameter_file.empty()) throw std::runtime_error("No parameter file specified.");
      // parse parameter file and CLI, log it to file
      json = JSONValue(parameter_file);

      for (const auto &param : cli_parameters) {
        std::string path, value, buf;

        // separate the str into a path to an entry and a value
        std::istringstream ss1(param.first);
        std::getline(ss1, path, '=');
        std::getline(ss1, value, '=');

        if (param.second == "double")
          json().at_pointer(path) = std::stod(value);
        else if (param.second == "bool")
          json().at_pointer(path) = (value == "true");
        else if (param.second == "int")
          json().at_pointer(path) = std::stoi(value);
        else if (param.second == "string")
          json().at_pointer(path) = value;
      }
    } catch (const std::exception &e) {
      std::cerr << "While reading the parameter file an error occured:\n    " << e.what() << "\n" << std::endl;
      print_usage_message();
      exit(0);
    }
  }

  void ConfigurationHelper::append(std::string str)
  {
    std::istringstream ss(str);
    std::string argv;
    while (std::getline(ss, argv, ' '))
      args.emplace_back(argv);
    try {
      parse_cli();
    } catch (const std::exception &e) {
      std::cerr << "While reading the CLI arguments an error occured:\n    " << e.what() << "\n" << std::endl;
      print_usage_message();
      exit(0);
    }
  }

  void ConfigurationHelper::print_usage_message()
  {
    std::string help_text = R"(
  ╭━━━━╮ ╭━━━╮╭━┳━━━━┳━━━━╮
  ╰╮╭╮ ┃ ┃╭━━╯┃╭┫╭━━╮┃╭━╮ ┃
   ┃┃┃ ┣━┫╰━━┳╯╰┫╰━━╯┃┃ ╰━╯
   ┃┃┃ ┣━┫╭━━┻╮╭┫╭╮ ╭┫┃╭━━╮
  ╭╯╰╯ ┃ ┃┃for┃┃┃┃┃ ╰┫╰┻━━┃
  ╰━━━━┻━┻╯   ╰╯╰╯╰━━┻━━━━╯
    The Discretisation Framework for functional Renormalisation Group flows.

This is a DiFfRG simulation. You can pass the following optional parameters to this executable:
  --help                      shows this text
  --generate-parameter-file   generates a parameter file with some default values
  -p                          specifiy a parameter file other than the standard parameter.json
  -sd                         overwrite a double parameter. This should be in the format '-s physical/T=0.1'
  -si                         overwrite an integer parameter. This should be in the format '-s physical/Nc=1'
  -sb                         overwrite a boolean parameter. This should be in the format '-s physical/use_sth=true'
  -ss                         overwrite a string parameter. This should be in the format '-s physical/a=hello'
)";
    std::cout << help_text << std::endl;
  }

  void ConfigurationHelper::parse_cli()
  {
    while (args.size()) {
      if (args.front() == std::string("--generate-parameter-file")) {
        generate_parameter_file();
        exit(0);
      } else if (args.front() == std::string("--help")) {
        print_usage_message();
        exit(0);
      } else if (args.front() == std::string("-p")) {
        if (args.size() == 1)
          throw std::runtime_error("Error: flag '-p' must be followed by the name of a parameter file.");
        args.pop_front();
        parameter_file = args.front();
        args.pop_front();
      } else if (args.front() == std::string("-sd")) {
        if (args.size() == 1) throw std::runtime_error("Error: flag '-sd' must be followed by a valid parameter.");
        args.pop_front();
        cli_parameters.push_back({args.front(), "double"});
        args.pop_front();
      } else if (args.front() == std::string("-si")) {
        if (args.size() == 1) throw std::runtime_error("Error: flag '-si' must be followed by a valid parameter.");
        args.pop_front();
        cli_parameters.push_back({args.front(), "int"});
        args.pop_front();
      } else if (args.front() == std::string("-sb")) {
        if (args.size() == 1) throw std::runtime_error("Error: flag '-sb' must be followed by a valid parameter.");
        args.pop_front();
        cli_parameters.push_back({args.front(), "bool"});
        args.pop_front();
      } else if (args.front() == std::string("-ss")) {
        if (args.size() == 1) throw std::runtime_error("Error: flag '-ss' must be followed by a valid parameter.");
        args.pop_front();
        cli_parameters.push_back({args.front(), "string"});
        args.pop_front();
      } else {
        std::cout << "Discarded CLI argument:    " << args.front() << std::endl;
        args.pop_front();
      }
    }
  }

  void ConfigurationHelper::generate_parameter_file()
  {
    JSONValue json = json::value(
        {{"physical", {}},
         {"integration",
          {{"x_quadrature_order", 32},
           {"angle_quadrature_order", 8},
           {"x0_quadrature_order", 16},
           {"x0_summands", 8},
           {"q0_quadrature_order", 16},
           {"q0_summands", 8},
           {"x_extent_tolerance", 1e-5},
           {"x0_extent_tolerance", 1e-5},
           {"q0_extent_tolerance", 1e-6},
           {"jacobian_quadrature_factor", 0.5}}},
         {"discretization",
          {{"fe_order", 3},
           {"threads", 8},
           {"batch_size", 16},
           {"overintegration", 0},
           {"output_subdivisions", 2},

           {"EoM_abs_tol", 1e-12},
           {"EoM_max_iter", 100},

           {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
           {"adaptivity",
            {{"start_adapt_at", 0.},
             {"adapt_dt", 1e-1},
             {"level", 0},
             {"refine_percent", 1e-1},
             {"coarsen_percent", 5e-2}}}}},
         {"timestepping",
          {{"final_time", 1.},
           {"output_dt", 1e-1},
           {"explicit",
            {{"dt", 1e-2}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-4}, {"rel_tol", 1e-4}}},
           {"implicit",
            {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-13}, {"rel_tol", 1e-7}}}}},
         {"output", {{"verbosity", 0}, {"folder", "output/"}, {"name", "output"}}}});

    std::ofstream file(parameter_file);
    json.print(file);
    file.close();
  }

  std::string ConfigurationHelper::get_log_file() const
  {
    std::string log_file = get_output_name() + ".log";
    return log_file;
  }
  std::string ConfigurationHelper::get_parameter_file() const { return parameter_file; }
  std::string ConfigurationHelper::get_output_name() const
  {
    std::string output_name = json.get_string("/output/name");
    return output_name;
  }
  std::string ConfigurationHelper::get_output_folder() const
  {
    std::string output_folder = make_folder(get_output_name());
    return output_folder;
  }
  std::string ConfigurationHelper::get_top_folder() const
  {
    std::string top_folder = json.get_string("/output/folder");
    top_folder = make_folder(top_folder);
    return top_folder;
  }

} // namespace DiFfRG