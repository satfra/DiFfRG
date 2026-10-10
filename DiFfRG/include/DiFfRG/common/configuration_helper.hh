#pragma once

// standard library
#include <list>
#include <string>
#include <utility>
#include <vector>

// DiFfRG
#include <DiFfRG/common/config_tree.hh>

// clang-format off
/**
 * @file configuration_helper.hh
 * @brief Reading the parameter file and the command line of a simulation.
 *
 * ### Command line
 * | Flag                              | Effect                                                                  |
 * |-----------------------------------|-------------------------------------------------------------------------|
 * | `--help`                          | print the usage message and exit                                        |
 * | `--generate-parameter-file`       | write a parameter file with default values and exit (see below)         |
 * | `-p <file>`                       | read `<file>` instead of the default parameter file                     |
 * | `-sd <key>=<value>`               | set a number, e.g. `-sd /physical/T=0.1`                                |
 * | `-si <key>=<value>`               | set an integer, e.g. `-si /integration/x_order=64`                      |
 * | `-sb <key>=<value>`               | set a boolean; `true` gives true, anything else false                   |
 * | `-ss <key>=<value>`               | set a string, e.g. `-ss /output/name=run1`                              |
 * | `--restart <snapshot>`            | continue from a snapshot file (see below)                               |
 * | `--snapshots-k <k1,k2,...>`       | write snapshots at these RG scales: `/timestepping/snapshots/k`         |
 * | `--snapshots-t <t1,t2,...>`       | write snapshots at these RG times: `/timestepping/snapshots/t`          |
 * | `--stop-after-last-snapshot`      | end the flow after the last snapshot                                    |
 *
 * Flags are read left to right; a later override of the same key wins. Unknown flags are ignored with a
 * warning.
 *
 * - `-sd`, `-si`, `-sb`, `-ss` only change keys that exist in the parameter file; a missing key is an error.
 *   They replace the entry including its type: `-sd` on an integer entry stores a floating-point number,
 *   which integer getters then no longer read (see config_tree.hh). Use `-si` for integer entries.
 * - The snapshot flags create their keys if needed.
 * - `--generate-parameter-file` writes to the parameter file name as known at that point, so put `-p` before
 *   it. An existing file of that name is overwritten. JSON or TOML is chosen from the file extension.
 * - With `--restart` the configuration is taken from the snapshot and no parameter file is read; `-p` is then
 *   an error, overrides with `-sd` etc. still apply. `/restart/file` is set to the snapshot.
 *
 * ### Default parameter file
 * Without `-p` the file name passed by the application is read (usually `parameter.json`). If it does not
 * exist but the same name with extension `.toml` does, that file is read instead.
 *
 * ### Errors
 * On a malformed command line or parameter file the constructor prints the error and the usage message and
 * terminates the process with exit code 1. `--help` and `--generate-parameter-file` terminate with exit code 0.
 */
// clang-format on

namespace DiFfRG
{
  /**
   * @brief Reads the parameter file and applies the command-line overrides.
   *
   * See configuration_helper.hh for the command-line flags. The usual way to obtain one is
   * DiFfRG::Init::get_configuration_helper().
   */
  class ConfigurationHelper
  {
  public:
    /**
     * @brief Parse the command line and read the parameter file.
     *
     * Everything happens here: the flags are parsed, the file is read and the overrides are applied. On
     * errors, `--help` and `--generate-parameter-file` the process is terminated, see configuration_helper.hh.
     *
     * @param argc Forwarded argc from main().
     * @param argv Forwarded argv from main().
     * @param parameter_file The default parameter file. If it does not exist, but the same name with extension
     * `.toml` does, that one is read. A file given with `-p` replaces it and is never substituted.
     */
    ConfigurationHelper(int argc, char *argv[], const std::string parameter_file = "parameter.json");

    /**
     * @brief Copy an unparsed helper.
     *
     * @throws std::runtime_error if `other` has already read its configuration, which is the case for every
     * helper built by the public constructors.
     */
    ConfigurationHelper(const ConfigurationHelper &other);

    /**
     * @brief Wrap an existing configuration; no command line and no file are read.
     *
     * @param config The configuration.
     */
    ConfigurationHelper(const ConfigTree &config);

    /**
     * @brief Parse further command-line arguments, given as one space-separated string.
     *
     * The configuration has already been read by the constructor, so `-sd`, `-si`, `-sb`, `-ss`, `-p` and the
     * snapshot flags given here do not change get_config().
     *
     * @param arg The arguments, e.g. "-sd /physical/T=0.1".
     */
    void append(std::string arg);

    /// The configuration, with all command-line overrides applied.
    ConfigTree &get_config();

    /// The configuration, with all command-line overrides applied.
    const ConfigTree &get_config() const;

    /// Old name of get_config().
    [[deprecated("Use get_config() instead")]] ConfigTree &get_json();

    /// Old name of get_config().
    [[deprecated("Use get_config() instead")]] const ConfigTree &get_json() const;

    /// Name of the log file. Deprecated, see OutputPath.
    [[deprecated("Use OutputPath(config).run_file(\".log\").filename().string() instead")]] std::string
    get_log_file() const;

    /**
     * @brief The parameter file that was read.
     *
     * This is the name after the `.toml` substitution; empty for a helper built from a ConfigTree. After
     * `--restart` no file is read and the default name is returned.
     */
    std::string get_parameter_file() const;

    /// Name of the run. Deprecated, see OutputPath.
    [[deprecated("Use OutputPath(config).run_name() instead")]] std::string get_output_name() const;
    /// Folder of the field output, with a trailing `/`. Deprecated, see OutputPath.
    [[deprecated("Use OutputPath(config).field_directory() instead")]] std::string get_output_folder() const;
    /// Top-level output folder, with a trailing `/`. Deprecated, see OutputPath.
    [[deprecated("Use OutputPath(config).root() instead")]] std::string get_top_folder() const;

    /**
     * @brief Read the configuration without printing anything or terminating.
     *
     * Same as the (argc, argv, parameter_file) constructor, including `-p`, the `.toml` substitution, the
     * overrides, `--restart` and the snapshot flags, but `--help` and `--generate-parameter-file` are skipped,
     * unknown flags are not reported, and any error yields an empty tree. DiFfRG::Init uses it to read a few
     * parameters before MPI and Kokkos are started.
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
