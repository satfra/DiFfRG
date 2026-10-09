#pragma once

// standard library
#include <list>
#include <string>
#include <utility>
#include <vector>

// DiFfRG
#include <DiFfRG/common/config_tree.hh>

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
     * @param parameter_file The name of the parameter file to be read. If this default name
     * does not exist on disk, but the same name with a ".toml" extension does, the latter
     * is read instead. A file given explicitly with the -p flag is never substituted.
     */
    ConfigurationHelper(int argc, char *argv[], const std::string parameter_file = "parameter.json");

    /**
     * @brief Copy a Configuration Helper
     */
    ConfigurationHelper(const ConfigurationHelper &);

    /**
     * @brief Construct a new Configuration Helper object from a ConfigTree object
     */
    ConfigurationHelper(const ConfigTree &config);

    /**
     * @brief Add a CLI-like argument to the list of arguments
     */
    void append(std::string arg);

    /**
     * @brief Obtain the configuration tree containing all the parameters
     */
    ConfigTree &get_config();

    /**
     * @brief Obtain the configuration tree containing all the parameters
     */
    const ConfigTree &get_config() const;

    /**
     * @brief Obtain the configuration tree containing all the parameters.
     *
     * Kept as the historical spelling of get_config().
     */
    [[deprecated("Use get_config() instead")]] ConfigTree &get_json();

    /**
     * @brief Obtain the configuration tree containing all the parameters.
     *
     * Kept as the historical spelling of get_config().
     */
    [[deprecated("Use get_config() instead")]] const ConfigTree &get_json() const;

    [[deprecated("Use OutputPath(config).run_file(\".log\").filename().string() instead")]] std::string
    get_log_file() const;
    std::string get_parameter_file() const;
    [[deprecated("Use OutputPath(config).run_name() instead")]] std::string get_output_name() const;
    [[deprecated("Use OutputPath(config).field_directory() instead")]] std::string get_output_folder() const;
    [[deprecated("Use OutputPath(config).root() instead")]] std::string get_top_folder() const;

    /**
     * @brief Read the configuration without producing any diagnostics or terminating.
     *
     * Behaves like the (argc, argv, parameter_file) constructor - same -p handling, same TOML
     * fallback, same -sd/-si/-sb/-ss overrides - but never prints a usage message, never warns
     * about unknown arguments and never calls exit(). Errors yield an empty tree instead.
     *
     * This exists so that DiFfRG::Init can look at a few parameters (notably
     * /discretization/threads) before the libraries are brought up, while leaving the
     * authoritative, diagnostics-producing parse to the application itself.
     */
    static ConfigTree probe(int argc, char *argv[], const std::string parameter_file = "parameter.json");

  private:
    ConfigurationHelper() = default;

    void parse();
    void parse_cli();
    void print_usage_message();
    void generate_parameter_file();
    /// The snapshot's configuration, without the seed run's own /restart.
    ConfigTree read_restart_configuration() const;
    void print_restart_banner(const std::vector<std::string> &overrides, bool continuing) const;

    std::list<std::string> args;
    std::list<std::pair<std::string, std::string>> cli_parameters;
    /// Set by the snapshot flags; unlike -sd & co. these create their key if the parameter file lacks it.
    std::list<std::pair<std::string, json::value>> cli_created_parameters;
    /// Set by --restart: the snapshot whose configuration replaces the parameter file.
    std::string restart_file;

    ConfigTree config;

    bool parsed = false;

    std::string parameter_file;

    /// Whether parameter_file was named explicitly with -p, which disables the fallback to
    /// a TOML file of the same name.
    bool parameter_file_explicit = false;

    /// In quiet mode nothing is printed and nothing exits the process; failures are thrown
    /// instead. Set by probe(). See probe() for why.
    bool quiet = false;
  };
} // namespace DiFfRG
