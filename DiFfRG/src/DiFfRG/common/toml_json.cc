// DiFfRG
#include "toml_json.hh"

// standard library
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

// external libraries
#include <toml11/toml.hpp>

namespace DiFfRG
{
  namespace internal
  {
    namespace
    {
      // The ordered type config keeps the key order of the source document, which matters
      // both for the readability of generated parameter files and for the json log written
      // alongside every run.
      using toml_value = toml::ordered_value;

      json::value toml_to_json(const toml_value &v)
      {
        switch (v.type()) {
        case toml::value_t::table: {
          json::object obj;
          for (const auto &[key, val] : v.as_table())
            obj.emplace(key, toml_to_json(val));
          return obj;
        }

        case toml::value_t::array: {
          json::array arr;
          for (const auto &val : v.as_array())
            arr.emplace_back(toml_to_json(val));
          return arr;
        }

        case toml::value_t::integer:
          return json::value(v.as_integer());
        case toml::value_t::floating:
          return json::value(v.as_floating());
        case toml::value_t::boolean:
          return json::value(v.as_boolean());
        case toml::value_t::string:
          return json::value(v.as_string());

        // TOML's date and time types have no json counterpart; keep their textual form.
        case toml::value_t::local_date: {
          std::ostringstream ss;
          ss << v.as_local_date();
          return json::value(ss.str());
        }
        case toml::value_t::local_time: {
          std::ostringstream ss;
          ss << v.as_local_time();
          return json::value(ss.str());
        }
        case toml::value_t::local_datetime: {
          std::ostringstream ss;
          ss << v.as_local_datetime();
          return json::value(ss.str());
        }
        case toml::value_t::offset_datetime: {
          std::ostringstream ss;
          ss << v.as_offset_datetime();
          return json::value(ss.str());
        }

        case toml::value_t::empty:
        default:
          return json::value(nullptr);
        }
      }

      /**
       * @brief The smallest precision that reproduces x exactly when read back.
       *
       * 17 significant digits always round-trip a double, but writing every number that
       * way turns 0.65 into 0.65000000000000002. Probing upwards from 6 digits instead
       * keeps hand-written values looking the way they were typed. This mirrors how toml11
       * serializes a defaultfloat value, so the precision computed here is the precision it
       * will actually use.
       */
      std::size_t shortest_round_trip_precision(const double x)
      {
        for (std::size_t prec = 6; prec < 17; ++prec) {
          std::ostringstream out;
          out << std::setprecision(static_cast<int>(prec)) << x;
          double back = 0.;
          std::istringstream in(out.str());
          in >> back;
          if (!in.fail() && back == x) return prec;
        }
        return 17;
      }

      /**
       * @brief Convert a json value to a toml value.
       *
       * @return false if the value is a null, which TOML cannot express and which the
       * caller therefore has to drop.
       */
      bool json_to_toml(const json::value &v, toml_value &out)
      {
        switch (v.kind()) {
        case json::kind::object: {
          toml::ordered_table table;
          for (const auto &[key, val] : v.get_object()) {
            toml_value entry;
            if (json_to_toml(val, entry)) table.emplace(std::string(key), std::move(entry));
          }
          out = toml_value(std::move(table));
          return true;
        }

        case json::kind::array: {
          toml::ordered_array arr;
          for (const auto &val : v.get_array()) {
            toml_value entry;
            if (json_to_toml(val, entry)) arr.emplace_back(std::move(entry));
          }
          // An empty array has no elements to infer a layout from, and toml11 then defaults
          // to the array-of-tables layout, writing a bare [[key]] header that reads back as
          // an array holding one empty table. Pin it to the inline form.
          toml::array_format_info fmt;
          if (arr.empty()) fmt.fmt = toml::array_format::oneline;
          out = toml_value(std::move(arr), fmt);
          return true;
        }

        case json::kind::string:
          out = toml_value(std::string(v.get_string()));
          return true;

        case json::kind::int64:
          out = toml_value(static_cast<std::int64_t>(v.get_int64()));
          return true;

        case json::kind::uint64:
          out = toml_value(static_cast<std::int64_t>(v.get_uint64()));
          return true;

        case json::kind::double_: {
          // defaultfloat keeps the written value close to how a human would have typed it,
          // and toml11 appends the ".0" needed to keep it a TOML float rather than an
          // integer.
          const double x = v.get_double();
          toml::floating_format_info fmt;
          fmt.fmt = toml::floating_format::defaultfloat;
          fmt.prec = shortest_round_trip_precision(x);
          out = toml_value(x, fmt);
          return true;
        }

        case json::kind::bool_:
          out = toml_value(v.get_bool());
          return true;

        case json::kind::null:
        default:
          return false;
        }
      }
    } // namespace

    json::value toml_file_to_json(const std::string &filename)
    {
      std::ifstream file(filename, std::ios::binary);
      if (!file.is_open()) throw std::runtime_error("Could not open file: " + filename);
      try {
        return toml_to_json(toml::parse<toml::ordered_type_config>(file, filename));
      } catch (const std::exception &e) {
        throw std::runtime_error("Failed to parse TOML file '" + filename + "': " + e.what());
      }
    }

    json::value toml_string_to_json(const std::string &content)
    {
      // Parsed through a stream rather than with parse_str, so that the diagnostic points
      // at the document instead of at the source location inside this file.
      std::istringstream stream(content);
      try {
        return toml_to_json(toml::parse<toml::ordered_type_config>(stream, "<string>"));
      } catch (const std::exception &e) {
        throw std::runtime_error("Failed to parse TOML: " + std::string(e.what()));
      }
    }

    std::string json_to_toml_string(const json::value &v)
    {
      if (!v.is_object())
        throw std::runtime_error("Cannot serialize as TOML: the top level of a TOML document "
                                 "must be a table.");
      toml_value root;
      json_to_toml(v, root);
      try {
        return toml::format(root);
      } catch (const std::exception &e) {
        throw std::runtime_error("Failed to serialize as TOML: " + std::string(e.what()));
      }
    }
  } // namespace internal
} // namespace DiFfRG
