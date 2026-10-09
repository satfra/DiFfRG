// standard library
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <ostream>
#include <set>
#include <sstream>

// DiFfRG
#include "toml_json.hh"
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/common/math.hh>

namespace DiFfRG
{
  namespace internal
  {
    bool obj_equal(const json::value &v1, const json::value &v2)
    {
      if (v1.kind() != v2.kind()) return false;

      if (v1.is_object()) {
        if (!v2.is_object()) return false;
        const auto &obj1 = v1.get_object();
        const auto &obj2 = v2.get_object();
        if (obj1.size() != obj2.size()) return false;
        for (const auto &[key, val] : obj1) {
          if (!obj2.contains(key)) return false;
          if (!obj_equal(val, obj2.at(key))) return false;
        }
        return true;
      }

      if (v1.is_array()) {
        if (!v2.is_array()) return false;
        const auto &arr1 = v1.get_array();
        const auto &arr2 = v2.get_array();
        if (arr1.size() != arr2.size()) return false;
        for (size_t i = 0; i < arr1.size(); ++i) {
          if (!obj_equal(arr1[i], arr2[i])) return false;
        }
        return true;
      }

      switch (v1.kind()) {
      case json::kind::string:
        return v1.get_string() == v2.get_string();
      case json::kind::uint64:
        return v1.get_uint64() == v2.get_uint64();
      case json::kind::int64:
        return v1.get_int64() == v2.get_int64();
      case json::kind::double_:
        return is_close(v1.get_double(), v2.get_double(), 1e-10);
      case json::kind::bool_:
        return v1.get_bool() == v2.get_bool();
      case json::kind::null:
        return true;
      default:
        return false;
      }

      return false;
    };
    bool has_toml_extension(const std::string &filename)
    {
      std::string extension = std::filesystem::path(filename).extension().string();
      std::transform(extension.begin(), extension.end(), extension.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      return extension == ".toml" || extension == ".tml";
    }

    /**
     * @brief Read a numeric node as a double, accepting integers as well.
     *
     * JSON and TOML both distinguish `2` from `2.0`; requiring the latter everywhere is a
     * trap for hand-written parameter files, so integers are widened here.
     */
    double as_double_relaxed(const json::value &v)
    {
      switch (v.kind()) {
      case json::kind::double_:
        return v.get_double();
      case json::kind::int64:
        return static_cast<double>(v.get_int64());
      case json::kind::uint64:
        return static_cast<double>(v.get_uint64());
      default:
        return v.as_double(); // throws with boost's own diagnostic
      }
    }
  } // namespace internal

  ConfigTree::ConfigTree() : value(json::object()) {}
  ConfigTree::ConfigTree(const std::string filename) : value(json::object())
  {
    // Anything that is not explicitly marked as TOML is read as JSON, so that files
    // without an extension keep their historical behaviour.
    if (internal::has_toml_extension(filename)) {
      value = internal::toml_file_to_json(filename);
      return;
    }

    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + filename);
    try {
      value = json::parse(file);
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to parse JSON file '" + filename + "': " + e.what());
    }
  }
  ConfigTree::ConfigTree(const char *filename) : ConfigTree(std::string(filename)) {}
  ConfigTree::ConfigTree(const json::value &v) : value(v) {}

  ConfigTree ConfigTree::from_json_string(const std::string &content)
  {
    try {
      return ConfigTree(json::parse(content));
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to parse JSON: " + std::string(e.what()));
    }
  }

  ConfigTree ConfigTree::from_toml_string(const std::string &content)
  {
    return ConfigTree(internal::toml_string_to_json(content));
  }

  json::value &ConfigTree::operator()() { return value; }

  ConfigTree::operator json::value() const { return value; }

  bool ConfigTree::operator==(const ConfigTree &other) const { return internal::obj_equal(value, other.value); }

  double ConfigTree::get_double(const std::string &key) const
  {
    try {
      return internal::as_double_relaxed(value.at_pointer(key));
    } catch (const std::exception &e) {
      throw std::runtime_error("ConfigTree::get_double error: " + std::string(e.what()) + "\n  At Key: " + key);
    }
  }

  namespace internal
  {
    /**
     * @brief Read an integer regardless of which arm boost::json stored it in.
     *
     * boost::json keeps a signed C++ integer as int64 and an unsigned one as uint64, and
     * `as_int64()` throws on the uint64 arm. A config parsed from a file therefore reads fine
     * while the same config built in C++ from an `unsigned` does not -- which made get_uint, the
     * accessor that exists precisely for unsigned values, unable to read them. The failure is
     * quiet where it matters: get_uint_or_warn catches the exception and falls back to its
     * default, so a caller silently gets the default value instead of the one it set.
     * as_double_relaxed above already handles both arms; these did not.
     */
    std::int64_t as_int64_relaxed(const boost::json::value &v)
    {
      if (v.is_uint64()) {
        const auto u = v.get_uint64();
        if (u > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
          throw std::runtime_error("value does not fit in a signed 64-bit integer");
        return static_cast<std::int64_t>(u);
      }
      return v.as_int64();
    }
  } // namespace internal

  int ConfigTree::get_int(const std::string &key) const
  {
    try {
      return internal::as_int64_relaxed(value.at_pointer(key));
    } catch (const std::exception &e) {
      throw std::runtime_error("ConfigTree::get_int error: " + std::string(e.what()) + "\n  At Key: " + key);
    }
  }

  uint ConfigTree::get_uint(const std::string &key) const
  {
    std::int64_t val;
    try {
      val = internal::as_int64_relaxed(value.at_pointer(key));
    } catch (const std::exception &e) {
      throw std::runtime_error("ConfigTree::get_uint error: " + std::string(e.what()) + "\n  At Key: " + key);
    }
    if (val < 0)
      throw std::runtime_error("ConfigTree::get_uint: value at key '" + key + "' is negative (" + std::to_string(val) +
                               ")");
    if (static_cast<std::uint64_t>(val) > std::numeric_limits<uint>::max())
      throw std::runtime_error("ConfigTree::get_uint: value at key '" + key + "' (" + std::to_string(val) +
                               ") does not fit in an unsigned int");
    return static_cast<uint>(val);
  }

  std::string ConfigTree::get_string(const std::string &key) const
  {
    try {
      return value.at_pointer(key).as_string().c_str();
    } catch (const std::exception &e) {
      throw std::runtime_error("ConfigTree::get_string error: " + std::string(e.what()) + "\n  At Key: " + key);
    }
  }

  bool ConfigTree::get_bool(const std::string &key) const
  {
    try {
      return value.at_pointer(key).as_bool();
    } catch (const std::exception &e) {
      throw std::runtime_error("ConfigTree::get_bool error: " + std::string(e.what()) + "\n  At Key: " + key);
    }
  }

  double ConfigTree::get_double(const std::string &key, const double def) const
  {
    try {
      return internal::as_double_relaxed(value.at_pointer(key));
    } catch (const std::exception &e) {
      return def;
    }
  }

  int ConfigTree::get_int(const std::string &key, const int def) const
  {
    // as_int64_relaxed, not as_int64: this overload swallows the exception and returns def, so
    // reading the wrong boost::json arm here is not an error the caller ever sees -- it is a
    // silent substitution of the default for the value they set. get_*_or_warn delegates here,
    // so this is the path that actually bites.
    try {
      return internal::as_int64_relaxed(value.at_pointer(key));
    } catch (const std::exception &e) {
      return def;
    }
  }

  uint ConfigTree::get_uint(const std::string &key, const uint def) const
  {
    // See the note on get_int(key, def) above.
    try {
      const std::int64_t val = internal::as_int64_relaxed(value.at_pointer(key));
      if (val < 0 || static_cast<std::uint64_t>(val) > std::numeric_limits<uint>::max())
        throw std::runtime_error("value is negative or does not fit in an unsigned int");
      return static_cast<uint>(val);
    } catch (const std::exception &e) {
      return def;
    }
  }

  std::string ConfigTree::get_string(const std::string &key, const std::string &def) const
  {
    try {
      return value.at_pointer(key).as_string().c_str();
    } catch (const std::exception &e) {
      return def;
    }
  }

  bool ConfigTree::get_bool(const std::string &key, const bool def) const
  {
    try {
      return value.at_pointer(key).as_bool();
    } catch (const std::exception &e) {
      return def;
    }
  }

  bool ConfigTree::contains(const std::string &key) const
  {
    try {
      (void)value.at_pointer(key);
      return true;
    } catch (const std::exception &) {
      return false;
    }
  }

  namespace internal
  {
    /**
     * @brief Report that a key is missing and a guessed default is being used.
     *
     * Kept to one message per key and process, so that reading the same parameter from many
     * integrators or assemblers does not flood the log.
     */
    void warn_missing_key(const std::string &key, const std::string &def)
    {
      static std::mutex mutex;
      static std::set<std::string> already_warned;

      {
        const std::lock_guard<std::mutex> lock(mutex);
        if (!already_warned.insert(key).second) return;
      }

      std::cerr << "WARNING: configuration key '" << key << "' is missing; falling back to " << def
                << ". This default is a guess, not a safe choice - set the key explicitly in your parameter file."
                << std::endl;
    }

    std::string format_double(double value)
    {
      std::stringstream ss;
      ss << std::scientific << std::setprecision(2) << value;
      return ss.str();
    }
  } // namespace internal

  double ConfigTree::get_double_or_warn(const std::string &key, const double def) const
  {
    if (contains(key)) return get_double(key, def);
    internal::warn_missing_key(key, internal::format_double(def));
    return def;
  }

  int ConfigTree::get_int_or_warn(const std::string &key, const int def) const
  {
    if (contains(key)) return get_int(key, def);
    internal::warn_missing_key(key, std::to_string(def));
    return def;
  }

  uint ConfigTree::get_uint_or_warn(const std::string &key, const uint def) const
  {
    if (contains(key)) return get_uint(key, def);
    internal::warn_missing_key(key, std::to_string(def));
    return def;
  }

  std::string ConfigTree::get_string_or_warn(const std::string &key, const std::string &def) const
  {
    if (contains(key)) return get_string(key, def);
    internal::warn_missing_key(key, "'" + def + "'");
    return def;
  }

  bool ConfigTree::get_bool_or_warn(const std::string &key, const bool def) const
  {
    if (contains(key)) return get_bool(key, def);
    internal::warn_missing_key(key, def ? "true" : "false");
    return def;
  }

  void ConfigTree::set_double(const std::string &key, double value)
  {
    try {
      this->value.at_pointer(key) = value;
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "ConfigTree::set_double error (missing key?)" << std::endl;
      ss << "  At Key: " << key << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  void ConfigTree::set_int(const std::string &key, int value)
  {
    try {
      this->value.at_pointer(key) = value;
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "ConfigTree::set_int error (missing key?)" << std::endl;
      ss << "  At Key: " << key << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  void ConfigTree::set_uint(const std::string &key, uint value)
  {
    try {
      this->value.at_pointer(key) = int(value);
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "ConfigTree::set_uint error (missing key?)" << std::endl;
      ss << "  At Key: " << key << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  void ConfigTree::set_string(const std::string &key, const std::string &value)
  {
    try {
      this->value.at_pointer(key) = value;
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "ConfigTree::set_string error (missing key?)" << std::endl;
      ss << "  At Key: " << key << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  void ConfigTree::set_bool(const std::string &key, bool value)
  {
    try {
      this->value.at_pointer(key) = value;
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "ConfigTree::set_bool error (missing key?)" << std::endl;
      ss << "  At Key: " << key << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  void ConfigTree::print(std::ostream &os) const { pretty_print(os, value); }

  void ConfigTree::print_toml(std::ostream &os) const { os << internal::json_to_toml_string(value) << std::endl; }

  void ConfigTree::pretty_print(std::ostream &os, json::value const &jv, std::string *indent) const
  {
    std::string indent_;
    std::stringstream ss;
    // set precision to 16 digits, this is the maximum precision of double, so its hopefully enough
    ss << std::scientific << std::setprecision(16);
    if (!indent) indent = &indent_;
    switch (jv.kind()) {
    case json::kind::object: {
      ss << "{\n";
      indent->append(4, ' ');
      auto const &obj = jv.get_object();
      if (!obj.empty()) {
        auto it = obj.begin();
        for (;;) {
          ss << *indent << json::serialize(it->key()) << " : ";
          pretty_print(ss, it->value(), indent);
          if (++it == obj.end()) break;
          ss << ",\n";
        }
      }
      ss << "\n";
      indent->resize(indent->size() - 4);
      ss << *indent << "}";
      break;
    }

    case json::kind::array: {
      ss << "[\n";
      indent->append(4, ' ');
      auto const &arr = jv.get_array();
      if (!arr.empty()) {
        auto it = arr.begin();
        for (;;) {
          ss << *indent;
          pretty_print(ss, *it, indent);
          if (++it == arr.end()) break;
          ss << ",\n";
        }
      }
      ss << "\n";
      indent->resize(indent->size() - 4);
      ss << *indent << "]";
      break;
    }

    case json::kind::string: {
      ss << json::serialize(jv.get_string());
      break;
    }

    case json::kind::uint64:
      ss << jv.get_uint64();
      break;

    case json::kind::int64:
      ss << jv.get_int64();
      break;

    case json::kind::double_:
      ss << jv.get_double();
      break;

    case json::kind::bool_:
      if (jv.get_bool())
        ss << "true";
      else
        ss << "false";
      break;

    case json::kind::null:
      ss << "null";
      break;
    }

    if (indent->empty()) ss << "\n";

    os << ss.str();
  }
} // namespace DiFfRG