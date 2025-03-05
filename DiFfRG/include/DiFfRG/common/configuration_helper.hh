#pragma once

// standard library
#include <fstream>
#include <list>
#include <string>

// external libraries
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  /**
   * @brief Class to read parameters given from the command line and from a parameter file.
   */
  class ConfigurationHelper
  {
  public:
    /**
     * @brief Construct a new Configuration Helper object
     * To get an overview of the available arguments, just pass --help as an argument.
     *
     * @param argc Forwarded argc from main() function
     * @param argv Forwarded argv from main() function
     * @param parameter_file The name of the parameter file to be read
     */
    ConfigurationHelper(int argc, char *argv[], const std::string parameter_file = "parameter.json");

    /**
     * @brief Copy a Configuration Helper
     */
    ConfigurationHelper(const ConfigurationHelper &);

    /**
     * @brief Construct a new Configuration Helper object from a JSONValue object
     */
    ConfigurationHelper(const JSONValue &json);

    /**
     * @brief Add a CLI-like argument to the list of arguments
     */
    void append(std::string arg);

    /**
     * @brief Obtain the JSON object containing all the parameters
     */
    JSONValue &get_json();

    /**
     * @brief Obtain the JSON object containing all the parameters
     */
    const JSONValue &get_json() const;

    std::string get_log_file() const;
    std::string get_parameter_file() const;
    std::string get_output_name() const;
    std::string get_output_folder() const;
    std::string get_top_folder() const;

  private:
    void parse();
    void parse_cli();
    void setup_logging();
    void print_usage_message();
    void generate_parameter_file();

    std::list<std::string> args;
    std::list<std::pair<std::string, std::string>> cli_parameters;

    uint depth_console = 0;
    uint depth_file = 3;

    JSONValue json;

    bool parsed;

    std::string parameter_file;

    static bool logger_initialized;
  };
} // namespace DiFfRG