#pragma once

// Private to the DiFfRG library sources: the toml11 implementation is confined to
// toml_json.cc, which is compiled into its own static library. Only boost::json values and
// std::strings cross this boundary, so no toml11 header ever enters the public include
// graph.

// standard library
#include <string>

// external libraries
#include <boost/json.hpp>

namespace DiFfRG
{
  namespace internal
  {
    namespace json = boost::json;

    /**
     * @brief Read a TOML file and convert it into a json value.
     *
     * Tables become objects, arrays become arrays, integers/floats/booleans/strings map
     * onto their json counterparts, and the TOML date/time types are converted to their
     * textual representation. Key order is preserved.
     *
     * @throws std::runtime_error on I/O or syntax errors, carrying the toml11 diagnostic.
     */
    json::value toml_file_to_json(const std::string &filename);

    /**
     * @brief Parse the contents of a TOML document and convert it into a json value.
     *
     * @throws std::runtime_error on syntax errors, carrying the toml11 diagnostic.
     */
    json::value toml_string_to_json(const std::string &content);

    /**
     * @brief Serialize a json value as a TOML document.
     *
     * Doubles are written in scientific notation with 16 significant digits, matching the
     * precision of the json pretty printer. Null entries are skipped, as TOML has no null.
     *
     * @throws std::runtime_error if the value is not an object at the top level.
     */
    std::string json_to_toml_string(const json::value &v);
  } // namespace internal
} // namespace DiFfRG
