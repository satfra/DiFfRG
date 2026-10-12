#pragma once

// standard library
#include <string>

// external libraries
#include <boost/json.hpp>

// clang-format off
/**
 * @file config_tree.hh
 * @brief ConfigTree: the parameter tree of a simulation, read from a JSON or TOML file.
 *
 * ### Keys
 * A key is a path into the tree, written as a JSON pointer: `"/timestepping/output_dt"` addresses
 * `output_dt` inside the `timestepping` section. Array entries are addressed by index (`"/grid/0"`), the
 * empty key `""` is the whole tree. A `/` or `~` inside a key name is written as `~1` or `~0`.
 *
 * ### Reading parameters
 * Every getter comes in three forms. They differ in what happens when the key is missing; a present entry that is
 * not readable as the requested type (e.g. a string where a number is expected) always throws:
 *
 * | Form                          | missing key                   | unreadable value              |
 * |-------------------------------|-------------------------------|-------------------------------|
 * | `get_double(key)`             | throws `std::runtime_error`   | throws `std::runtime_error`   |
 * | `get_double(key, def)`        | returns `def`                 | throws `std::runtime_error`   |
 * | `get_double_or_warn(key, def)`| returns `def`, warns once     | throws `std::runtime_error`   |
 *
 * and likewise for `int`, `uint`, `string` and `bool`. Numbers are converted only in one direction: an
 * integer entry (`2`) can be read with `get_double`, but a floating-point entry (`2.0`) cannot be read with
 * `get_int` or `get_uint`. `get_uint` also rejects negative values.
 *
 * @code
 * const DiFfRG::ConfigTree config("parameter.json");
 * const double Lambda = config.get_double("/physical/Lambda");            // required
 * const double T = config.get_double("/physical/T", 0.);                  // optional, default 0
 * const uint order = config.get_uint_or_warn("/integration/x_order", 32); // default is a guess
 * @endcode
 *
 * ### Writing parameters
 * `set_*` overwrite an existing entry; they cannot create keys, and throw if the key (or one of its
 * parent sections) does not exist. The stored type is replaced by the type of the setter, e.g. `set_double`
 * on an integer entry turns it into a floating-point entry. To build a tree from scratch, construct it from a
 * `json::value`:
 * @code
 * DiFfRG::ConfigTree config(DiFfRG::json::value({{"physical", {{"Lambda", 1.}, {"T", 0.1}}}}));
 * @endcode
 *
 * ### File formats
 * Files ending in `.toml` or `.tml` (any case) are read as TOML, everything else as JSON. Internally the
 * tree is always a JSON value, so both formats are addressed by the same keys. JSON files may not contain
 * comments or trailing commas; TOML files may contain comments. TOML dates and times are stored as strings.
 */
// clang-format on

namespace DiFfRG
{
  /// Short name for Boost.JSON, whose `json::value` is the storage of a ConfigTree.
  namespace json = boost::json;

  namespace internal
  {
    /**
     * @brief Whether a file name denotes a TOML file, i.e. ends in ".toml" or ".tml".
     *
     * The comparison is case-insensitive. Everything else is treated as JSON.
     */
    bool has_toml_extension(const std::string &filename);
  } // namespace internal

  /**
   * @brief A hierarchical parameter tree, read from a JSON or TOML file.
   *
   * See config_tree.hh for the key syntax and for how the getters treat missing or mistyped entries.
   */
  class ConfigTree
  {
  public:
    /// An empty tree (an empty JSON object).
    ConfigTree();

    /**
     * @brief Read a tree from a file.
     *
     * Files ending in ".toml" or ".tml" are read as TOML, everything else as JSON.
     *
     * @param filename The name of the file to read.
     * @throws std::runtime_error if the file cannot be opened or parsed.
     */
    ConfigTree(const std::string filename);

    /**
     * @brief Read a tree from a file.
     *
     * Without this overload a string literal would be ambiguous between a file name and a json value.
     *
     * @param filename The name of the file to read.
     * @throws std::runtime_error if the file cannot be opened or parsed.
     */
    ConfigTree(const char *filename);

    /**
     * @brief Wrap a json value.
     *
     * @param v The json value to copy into the tree.
     */
    ConfigTree(const json::value &v);

    /**
     * @brief Parse a tree from the text of a JSON document.
     *
     * @param content The JSON document.
     * @throws std::runtime_error on a parse error.
     */
    static ConfigTree from_json_string(const std::string &content);

    /**
     * @brief Parse a tree from the text of a TOML document.
     *
     * @param content The TOML document.
     * @throws std::runtime_error on a parse error.
     */
    static ConfigTree from_toml_string(const std::string &content);

    /// Direct, modifiable access to the wrapped json value.
    json::value &operator()();

    /// A copy of the wrapped json value.
    operator json::value() const;

    /**
     * @brief Whether both trees hold the same keys and values.
     *
     * Floating-point entries are compared with a relative and absolute tolerance of \f$10^{-10}\f$. Entries must
     * have the same type: `2` and `2.0` differ, and so do a signed and an unsigned integer of the same value.
     */
    bool operator==(const ConfigTree &other) const;

    /**
     * @brief Read a number. Integer entries are accepted as well.
     *
     * @param key The key, e.g. "/physical/Lambda".
     * @throws std::runtime_error if the key is missing or the entry is not a number.
     */
    double get_double(const std::string &key) const;

    /**
     * @brief Read an integer.
     *
     * @param key The key, e.g. "/integration/x_order".
     * @throws std::runtime_error if the key is missing or the entry is not an integer (`2.0` is not).
     */
    int get_int(const std::string &key) const;

    /**
     * @brief Read a non-negative integer.
     *
     * @param key The key, e.g. "/integration/x_order".
     * @throws std::runtime_error if the key is missing, the entry is not an integer, or it is negative.
     */
    uint get_uint(const std::string &key) const;

    /**
     * @brief Read a string.
     *
     * @param key The key, e.g. "/output/name".
     * @throws std::runtime_error if the key is missing or the entry is not a string.
     */
    std::string get_string(const std::string &key) const;

    /**
     * @brief Read a boolean.
     *
     * @param key The key, e.g. "/output/vtk".
     * @throws std::runtime_error if the key is missing or the entry is not a boolean.
     */
    bool get_bool(const std::string &key) const;

    /**
     * @brief Read a number, or return `def`. Integer entries are accepted as well.
     *
     * @param key The key, e.g. "/physical/T".
     * @param def Returned if the key is missing; an entry that is not a number throws.
     */
    double get_double(const std::string &key, const double def) const;

    /**
     * @brief Read an integer, or return `def`.
     *
     * @param key The key.
     * @param def Returned if the key is missing; an entry that is not an integer (e.g. `2.0`) throws.
     */
    int get_int(const std::string &key, const int def) const;

    /**
     * @brief Read a non-negative integer, or return `def`.
     *
     * @param key The key.
     * @param def Returned if the key is missing; an entry that is not a non-negative integer throws.
     */
    uint get_uint(const std::string &key, const uint def) const;

    /**
     * @brief Read a string, or return `def`.
     *
     * @param key The key.
     * @param def Returned if the key is missing; an entry that is not a string throws.
     */
    std::string get_string(const std::string &key, const std::string &def) const;

    /**
     * @brief Read a boolean, or return `def`.
     *
     * @param key The key.
     * @param def Returned if the key is missing; an entry that is not a boolean throws.
     */
    bool get_bool(const std::string &key, const bool def) const;

    /**
     * @brief Whether the tree has an entry at `key`.
     *
     * @param key The key, e.g. "/physical/T".
     */
    bool contains(const std::string &key) const;

    /** @name Getters that warn when they fall back to the default
     *
     * Like the two-argument getters, but they print a warning to stderr (once per key and process) whenever
     * `def` is returned, i.e. when the key is missing. Use
     * them for parameters whose default is a guess rather than a safe choice: quadrature orders, solver
     * tolerances, grids, the cutoff scale.
     * @{
     */
    /// Read a number (integers accepted), or warn and return `def`.
    double get_double_or_warn(const std::string &key, const double def) const;
    /// Read an integer, or warn and return `def`.
    int get_int_or_warn(const std::string &key, const int def) const;
    /// Read a non-negative integer, or warn and return `def`.
    uint get_uint_or_warn(const std::string &key, const uint def) const;
    /// Read a string, or warn and return `def`.
    std::string get_string_or_warn(const std::string &key, const std::string &def) const;
    /// Read a boolean, or warn and return `def`.
    bool get_bool_or_warn(const std::string &key, const bool def) const;
    /** @} */

    /**
     * @brief Overwrite an existing entry with a number.
     *
     * @param key The key; it must already exist.
     * @param value The new value. The entry becomes a floating-point entry, whatever it was before.
     * @throws std::runtime_error if the key does not exist.
     */
    void set_double(const std::string &key, double value);

    /**
     * @brief Overwrite an existing entry with an integer.
     *
     * @param key The key; it must already exist.
     * @param value The new value.
     * @throws std::runtime_error if the key does not exist.
     */
    void set_int(const std::string &key, int value);

    /**
     * @brief Overwrite an existing entry with a non-negative integer.
     *
     * @param key The key; it must already exist.
     * @param value The new value.
     * @throws std::runtime_error if the key does not exist.
     */
    void set_uint(const std::string &key, uint value);

    /**
     * @brief Overwrite an existing entry with a string.
     *
     * @param key The key; it must already exist.
     * @param value The new value.
     * @throws std::runtime_error if the key does not exist.
     */
    void set_string(const std::string &key, const std::string &value);

    /**
     * @brief Overwrite an existing entry with a boolean.
     *
     * @param key The key; it must already exist.
     * @param value The new value.
     * @throws std::runtime_error if the key does not exist.
     */
    void set_bool(const std::string &key, bool value);

    /**
     * @brief Write the tree as indented JSON.
     *
     * @param os The stream to write to.
     */
    void print(std::ostream &os) const;

    /**
     * @brief Write the tree as TOML.
     *
     * @param os The stream to write to.
     * @throws std::runtime_error if the top level of the tree is not an object.
     */
    void print_toml(std::ostream &os) const;

  private:
    json::value value;

    void pretty_print(std::ostream &os, json::value const &v, std::string *indent = nullptr) const;
  };

  /**
   * @brief Deprecated former name of ConfigTree.
   */
  using JSONValue [[deprecated("JSONValue was renamed. Use DiFfRG::ConfigTree instead.")]] = ConfigTree;
} // namespace DiFfRG
