#pragma once

// standard library
#include <string>

// external libraries
#include <boost/json.hpp>

namespace DiFfRG
{
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
   * @brief A hierarchical configuration tree, readable from JSON and TOML files.
   *
   * This class allows to easily access and modify the values stored in a configuration
   * object. Internally the tree is held as a boost json value, independently of the file
   * format it was read from; keys are therefore always JSON pointers of the form
   * "/category/subcategory/.../key".
   */
  class ConfigTree
  {
  public:
    /**
     * @brief Construct an empty ConfigTree object
     *
     */
    ConfigTree();

    /**
     * @brief Construct a new ConfigTree object from a file.
     *
     * Files ending in ".toml" or ".tml" are read as TOML, everything else as JSON.
     *
     * @param filename The name of the file to read.
     */
    ConfigTree(const std::string filename);

    /**
     * @brief Construct a new ConfigTree object from a file.
     *
     * Without this overload a string literal is ambiguous between the file name and the
     * json value constructor, as boost json values are constructible from a char pointer.
     *
     * @param filename The name of the file to read.
     */
    ConfigTree(const char *filename);

    /**
     * @brief Construct a new ConfigTree object
     *
     * @param v The json value to wrap.
     */
    ConfigTree(const json::value &v);

    /**
     * @brief Construct a ConfigTree from the contents of a JSON document.
     *
     * @param content The JSON document.
     */
    static ConfigTree from_json_string(const std::string &content);

    /**
     * @brief Construct a ConfigTree from the contents of a TOML document.
     *
     * @param content The TOML document.
     */
    static ConfigTree from_toml_string(const std::string &content);

    /**
     * @brief The call operator gives access to the wrapped json value.
     */
    json::value &operator()();

    /**
     * @brief The call operator gives access to the wrapped json value.
     */
    operator json::value() const;

    /**
     * @brief Equality operator. Checks all values in the configuration object to be equal.
     */
    bool operator==(const ConfigTree &other) const;

    /**
     * @brief Get the value of a key in the configuration object.
     *
     * Integer entries are accepted and converted, so that both `2` and `2.0` can be read.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @return double The value.
     */
    double get_double(const std::string &key) const;

    /**
     * @brief Get the value of a key in the configuration object.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @return int The value.
     */
    int get_int(const std::string &key) const;

    /**
     * @brief Get the value of a key in the configuration object.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @return uint The value.
     */
    uint get_uint(const std::string &key) const;

    /**
     * @brief Get the value of a key in the configuration object.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @return std::string The value.
     */
    std::string get_string(const std::string &key) const;

    /**
     * @brief Get the value of a key in the configuration object.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @return bool The value.
     */
    bool get_bool(const std::string &key) const;

    /**
     * @brief Get the value of a key in the configuration object, with a default value.
     *
     * Integer entries are accepted and converted, so that both `2` and `2.0` can be read.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @param def The default value to return if the key is not found.
     * @return double The value.
     */
    double get_double(const std::string &key, const double def) const;

    /**
     * @brief Get the value of a key in the configuration object, with a default value.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @param def The default value to return if the key is not found.
     * @return int The value.
     */
    int get_int(const std::string &key, const int def) const;

    /**
     * @brief Get the value of a key in the configuration object, with a default value.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @param def The default value to return if the key is not found.
     * @return uint The value.
     */
    uint get_uint(const std::string &key, const uint def) const;

    /**
     * @brief Get the value of a key in the configuration object, with a default value.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @param def The default value to return if the key is not found.
     * @return std::string The value.
     */
    std::string get_string(const std::string &key, const std::string &def) const;

    /**
     * @brief Get the value of a key in the configuration object, with a default value.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @param def The default value to return if the key is not found.
     * @return bool The value.
     */
    bool get_bool(const std::string &key, const bool def) const;

    /**
     * @brief Whether the configuration object has an entry at the given key.
     *
     * @param key The key to look for. Format: "/category/subcategory/.../key".
     */
    bool contains(const std::string &key) const;

    /** @name Defaulted getters that warn when the default is used.
     *
     * These behave exactly like the two-argument getters above, but additionally print a
     * warning to stderr (once per key and process) whenever the fallback is taken. They are
     * meant for parameters whose default is a *guess* rather than a safe choice: quadrature
     * orders, solver tolerances, grids, the cutoff scale. A run that silently takes such a
     * default is likely producing results the user did not intend.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @param def The default value to return if the key is not found.
     * @{
     */
    double get_double_or_warn(const std::string &key, const double def) const;
    int get_int_or_warn(const std::string &key, const int def) const;
    uint get_uint_or_warn(const std::string &key, const uint def) const;
    std::string get_string_or_warn(const std::string &key, const std::string &def) const;
    bool get_bool_or_warn(const std::string &key, const bool def) const;
    /** @} */

    /**
     * @brief Set the value of a key in the configuration object.
     *
     * @param key The key to set the value for. Format: "/category/subcategory/.../key".
     * @param value The value to set.
     */
    void set_double(const std::string &key, double value);

    /**
     * @brief Set the value of a key in the configuration object.
     *
     * @param key The key to set the value for. Format: "/category/subcategory/.../key".
     * @param value The value to set.
     */
    void set_int(const std::string &key, int value);

    /**
     * @brief Set the value of a key in the configuration object.
     *
     * @param key The key to set the value for. Format: "/category/subcategory/.../key".
     * @param value The value to set.
     */
    void set_uint(const std::string &key, uint value);

    /**
     * @brief Set the value of a key in the configuration object.
     *
     * @param key The key to set the value for. Format: "/category/subcategory/.../key".
     * @param value The value to set.
     */
    void set_string(const std::string &key, const std::string &value);

    /**
     * @brief Set the value of a key in the configuration object.
     *
     * @param key The key to set the value for. Format: "/category/subcategory/.../key".
     * @param value The value to set.
     */
    void set_bool(const std::string &key, bool value);

    /**
     * @brief Pretty-print the configuration tree as JSON to a stream.
     */
    void print(std::ostream &os) const;

    /**
     * @brief Print the configuration tree as TOML to a stream.
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
