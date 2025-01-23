// standard library
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>

// DiFfRG
#include <DiFfRG/common/json.hh>
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
        if (v1.get_string() != v2.get_string())
          std::cout << "v1: " << v1.get_string() << " v2: " << v2.get_string() << std::endl;
        return v1.get_string() == v2.get_string();
      case json::kind::uint64:
        if (v1.get_uint64() != v2.get_uint64())
          std::cout << "v1: " << v1.get_uint64() << " v2: " << v2.get_uint64() << std::endl;
        return v1.get_uint64() == v2.get_uint64();
      case json::kind::int64:
        if (v1.get_int64() != v2.get_int64())
          std::cout << "v1: " << v1.get_int64() << " v2: " << v2.get_int64() << std::endl;
        return v1.get_int64() == v2.get_int64();
      case json::kind::double_:
        if (!is_close(v1.get_double(), v2.get_double(), 1e-10))
          std::cout << "v1: " << v1.get_double() << " v2: " << v2.get_double() << std::endl;
        return is_close(v1.get_double(), v2.get_double(), 1e-10);
      case json::kind::bool_:
        if (v1.get_bool() != v2.get_bool())
          std::cout << "v1: " << v1.get_bool() << " v2: " << v2.get_bool() << std::endl;
        return v1.get_bool() == v2.get_bool();
      case json::kind::null:
        return true;
      default:
        return false;
      }

      return false;
    };
  } // namespace internal

  JSONValue::JSONValue() : value(json::object()) {}
  JSONValue::JSONValue(const std::string filename) : value(json::object())
  {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + filename);
    value = json::parse(file);
  }
  JSONValue::JSONValue(const json::value &v) : value(v) {}

  json::value &JSONValue::operator()() { return value; }

  JSONValue::operator json::value() const { return value; }

  bool JSONValue::operator==(const JSONValue &other) const { return internal::obj_equal(value, other.value); }

  double JSONValue::get_double(const std::string &key) const
  {
    try {
      return value.at_pointer(key).as_double();
    } catch (const std::exception &e) {
      // stingstream
      std::stringstream ss;
      ss << "JSONValue::get_double error:" << std::endl;
      ss << "  At Key: " << key << std::endl;
      ss << "  Error: " << e.what() << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  int JSONValue::get_int(const std::string &key) const
  {
    try {
      return value.at_pointer(key).as_int64();
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "JSONValue::get_int error:" << std::endl;
      ss << "  At Key: " << key << std::endl;
      ss << "  Error: " << e.what() << std::endl << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  uint JSONValue::get_uint(const std::string &key) const
  {
    try {
      const int val = value.at_pointer(key).as_int64();
      return val >= 0 ? val : throw std::runtime_error("Value is negative");
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "JSONValue::get_uint error:" << std::endl;
      ss << "  At Key: " << key << std::endl;
      ss << "  Error: " << e.what() << std::endl << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  std::string JSONValue::get_string(const std::string &key) const
  {
    try {
      return value.at_pointer(key).as_string().c_str();
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "JSONValue::get_string error:" << std::endl;
      ss << "  At Key: " << key << std::endl;
      ss << "  Error: " << e.what() << std::endl << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  bool JSONValue::get_bool(const std::string &key) const
  {
    try {
      return value.at_pointer(key).as_bool();
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "JSONValue::get_bool error:" << std::endl;
      ss << "  At Key: " << key << std::endl;
      ss << "  Error: " << e.what() << std::endl << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  double JSONValue::get_double(const std::string &key, const double def) const
  {
    try {
      return value.at_pointer(key).as_double();
    } catch (const std::exception &e) {
      return def;
    }
    return value.at_pointer(key).as_double();
  }

  int JSONValue::get_int(const std::string &key, const int def) const
  {
    try {
      return value.at_pointer(key).as_int64();
    } catch (const std::exception &e) {
      return def;
    }
  }

  uint JSONValue::get_uint(const std::string &key, const uint def) const
  {
    try {
      const int val = value.at_pointer(key).as_int64();
      return val >= 0 ? val : throw std::runtime_error("Value is negative");
    } catch (const std::exception &e) {
      return def;
    }
  }

  std::string JSONValue::get_string(const std::string &key, const std::string &def) const
  {
    try {
      return value.at_pointer(key).as_string().c_str();
    } catch (const std::exception &e) {
      return def;
    }
  }

  bool JSONValue::get_bool(const std::string &key, const bool def) const
  {
    try {
      return value.at_pointer(key).as_bool();
    } catch (const std::exception &e) {
      return def;
    }
  }

  void JSONValue::set_double(const std::string &key, double value)
  {
    try {
      this->value.at_pointer(key) = value;
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "JSONValue::set_double error:" << std::endl;
      ss << "  At Key: " << key << std::endl;
      ss << "  Error: " << e.what() << std::endl << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  void JSONValue::set_int(const std::string &key, int value)
  {
    try {
      this->value.at_pointer(key) = value;
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "JSONValue::set_int error:" << std::endl;
      ss << "  At Key: " << key << std::endl;
      ss << "  Error: " << e.what() << std::endl << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  void JSONValue::set_uint(const std::string &key, uint value)
  {
    try {
      this->value.at_pointer(key) = int(value);
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "JSONValue::set_uint error:" << std::endl;
      ss << "  At Key: " << key << std::endl;
      ss << "  Error: " << e.what() << std::endl << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  void JSONValue::set_string(const std::string &key, const std::string &value)
  {
    try {
      this->value.at_pointer(key) = value;
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "JSONValue::set_string error:" << std::endl;
      ss << "  At Key: " << key << std::endl;
      ss << "  Error: " << e.what() << std::endl << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  void JSONValue::set_bool(const std::string &key, bool value)
  {
    try {
      this->value.at_pointer(key) = value;
    } catch (const std::exception &e) {
      std::stringstream ss;
      ss << "JSONValue::set_bool error:" << std::endl;
      ss << "  At Key: " << key << std::endl;
      ss << "  Error: " << e.what() << std::endl << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  void JSONValue::print(std::ostream &os) const { pretty_print(os, value); }

  void JSONValue::pretty_print(std::ostream &os, json::value const &jv, std::string *indent) const
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