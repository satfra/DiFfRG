#pragma once

// standard library
#include <concepts>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// external libraries
#include <deal.II/base/utilities.h>

// DiFfRG
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/common/fixed_string.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/tuples.hh>

namespace DiFfRG
{
  using uint = unsigned int;

  using dealii::Utilities::int_to_string;

  /**
   * @brief A compile-time for loop, which calls the lambda f of signature void(integer) for each index.
   */
  template <auto Start, auto End, auto Inc, class F> constexpr void constexpr_for(F &&f)
  {
    if constexpr (Start < End) {
      f(std::integral_constant<decltype(Start), Start>());
      constexpr_for<Start + Inc, End, Inc>(f);
    }
  }

  /**
   * @brief Strips all special characters from a string, e.g. for use in filenames.
   *
   * @param name The string to be stripped
   * @return std::string The stripped string
   */
  std::string strip_name(const std::string &name);

  /**
   * @brief Return number with fixed precision after the decimal point
   */
  template <typename T> std::string getWithPrecision(uint precision, T number)
  {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(precision) << number;
    return stream.str();
  }

  /**
   * @brief Checks if a file exists.
   *
   * @param name The name of the file.
   */
  bool file_exists(const std::string &name);

  /**
   * @brief Return number with fixed significant digits
   */
  template <typename T> std::string to_string_with_digits(const T number, const int digits)
  {
    if (digits <= 0) throw std::runtime_error("to_string_with_digits: digits must be > 0");
    const int d = int(std::ceil(std::log10(number < 0 ? -number : number)));
    std::stringstream stream;
    stream << std::fixed << std::setprecision(std::max(digits - d, 0)) << number;
    return stream.str();
  }

  /**
   * @brief Add a trailing '/' to a string, in order for it to be in standard form of a folder path.
   */
  std::string make_folder(const std::string &path);

  /**
   * @brief Creates the directory path, even if its parent directories should not exist.
   */
  bool create_folder(const std::string &path_);

  /**
   * @brief Nice output from seconds to h/min/s style string
   */
  std::string time_format(size_t time_in_seconds);

  template <typename T>
    requires(!std::is_same_v<T, size_t>)
  std::string time_format(T time_in_seconds)
  {
    return time_format(static_cast<size_t>(time_in_seconds));
  }

  /**
   * @brief Nice output from seconds to h/min/s style string
   */
  std::string time_format_ms(size_t time_in_miliseconds);

  bool has_suffix(const std::string &str, const std::string &suffix);
} // namespace DiFfRG
