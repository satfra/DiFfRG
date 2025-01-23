#pragma once

// standard library
#include <string>

// external libraries
#include <boost/json.hpp>

namespace DiFfRG
{
  namespace json = boost::json;

  /**
   * @brief A wrapper around the boost json value class.
   *
   * This class allows to easily access and modify the values stored in a json object.
   */
  class JSONValue
  {
  public:
    /**
     * @brief Construct an empty JSONValue object
     *
     */
    JSONValue();

    /**
     * @brief Construct a new JSONValue object from a json file.
     *
     * @param filename The name of the file to read.
     */
    JSONValue(const std::string filename);

    /**
     * @brief Construct a new JSONValue object
     *
     * @param v The json value to wrap.
     */
    JSONValue(const json::value &v);

    /**
     * @brief The call operator gives access to the wrapped json value.
     */
    json::value &operator()();

    /**
     * @brief The call operator gives access to the wrapped json value.
     */
    operator json::value() const;

    /**
     * @brief Equality operator. Checks all values in the json object to be equal.
     */
    bool operator==(const JSONValue &other) const;

    /**
     * @brief Get the value of a key in the json object.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @return double The value.
     */
    double get_double(const std::string &key) const;

    /**
     * @brief Get the value of a key in the json object.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @return int The value.
     */
    int get_int(const std::string &key) const;

    /**
     * @brief Get the value of a key in the json object.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @return uint The value.
     */
    uint get_uint(const std::string &key) const;

    /**
     * @brief Get the value of a key in the json object.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @return std::string The value.
     */
    std::string get_string(const std::string &key) const;

    /**
     * @brief Get the value of a key in the json object.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @return bool The value.
     */
    bool get_bool(const std::string &key) const;

    /**
     * @brief Get the value of a key in the json object, with a default value.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @param def The default value to return if the key is not found.
     * @return double The value.
     */
    double get_double(const std::string &key, const double def) const;

    /**
     * @brief Get the value of a key in the json object, with a default value.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @param def The default value to return if the key is not found.
     * @return int The value.
     */
    int get_int(const std::string &key, const int def) const;

    /**
     * @brief Get the value of a key in the json object, with a default value.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @param def The default value to return if the key is not found.
     * @return uint The value.
     */
    uint get_uint(const std::string &key, const uint def) const;

    /**
     * @brief Get the value of a key in the json object, with a default value.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @param def The default value to return if the key is not found.
     * @return std::string The value.
     */
    std::string get_string(const std::string &key, const std::string &def) const;

    /**
     * @brief Get the value of a key in the json object, with a default value.
     *
     * @param key The key to get the value for. Format: "/category/subcategory/.../key".
     * @param def The default value to return if the key is not found.
     * @return bool The value.
     */
    bool get_bool(const std::string &key, const bool def) const;

    /**
     * @brief Set the value of a key in the json object.
     *
     * @param key The key to set the value for. Format: "/category/subcategory/.../key".
     * @param value The value to set.
     */
    void set_double(const std::string &key, double value);

    /**
     * @brief Set the value of a key in the json object.
     *
     * @param key The key to set the value for. Format: "/category/subcategory/.../key".
     * @param value The value to set.
     */
    void set_int(const std::string &key, int value);

    /**
     * @brief Set the value of a key in the json object.
     *
     * @param key The key to set the value for. Format: "/category/subcategory/.../key".
     * @param value The value to set.
     */
    void set_uint(const std::string &key, uint value);

    /**
     * @brief Set the value of a key in the json object.
     *
     * @param key The key to set the value for. Format: "/category/subcategory/.../key".
     * @param value The value to set.
     */
    void set_string(const std::string &key, const std::string &value);

    /**
     * @brief Set the value of a key in the json object.
     *
     * @param key The key to set the value for. Format: "/category/subcategory/.../key".
     * @param value The value to set.
     */
    void set_bool(const std::string &key, bool value);

    /**
     * @brief Pretty-print the json value to a stream.
     */
    void print(std::ostream &os) const;

  private:
    json::value value;

    void pretty_print(std::ostream &os, json::value const &v, std::string *indent = nullptr) const;
  };
} // namespace DiFfRG