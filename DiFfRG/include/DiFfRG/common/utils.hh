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
   * @brief Compile-time loop: calls `f(i)` for `i = Start, Start + Inc, ...` while `i < End`.
   *
   * `i` is passed as a `std::integral_constant`, so it can be used as a template argument inside `f`. `Inc` must be
   * positive.
   * @code
   * constexpr_for<0, 3, 1>([&](auto i) { std::get<i>(tuple) = 0; });
   * @endcode
   */
  template <auto Start, auto End, auto Inc, class F> constexpr void constexpr_for(F &&f)
  {
    if constexpr (Start < End) {
      f(std::integral_constant<decltype(Start), Start>());
      constexpr_for<Start + Inc, End, Inc>(f);
    }
  }

  /**
   * @brief Remove spaces and punctuation from a string, e.g. for use in file names.
   *
   * Removes the characters `` ^$#%&*-+=~`!@?<>,.:;'"|\/(){}[]`` and spaces; letters, digits and `_` are kept.
   */
  std::string strip_name(const std::string &name);

  /// `number` printed with `precision` digits after the decimal point.
  template <typename T> std::string getWithPrecision(uint precision, T number)
  {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(precision) << number;
    return stream.str();
  }

  /// Whether the file `name` exists and can be opened for reading.
  bool file_exists(const std::string &name);

  /**
   * @brief `number` printed in fixed notation with (at least) `digits` significant digits.
   *
   * E.g. 3.14159 with 3 digits gives "3.14", 0.0123456 gives "0.0123". Numbers with more than `digits` digits before
   * the decimal point are printed without decimals, i.e. with more significant digits.
   */
  template <typename T> std::string to_string_with_digits(const T number, const int digits)
  {
    if (digits <= 0) throw std::runtime_error("to_string_with_digits: digits must be > 0");
    // Number of digits before the decimal point (zero or negative for |number| < 1).
    const int d = number == 0 ? 1 : int(std::floor(std::log10(number < 0 ? -number : number))) + 1;
    std::stringstream stream;
    stream << std::fixed << std::setprecision(std::max(digits - d, 0)) << number;
    return stream.str();
  }

  /// `path` with a trailing '/' added if it has none.
  std::string make_folder(const std::string &path);

  /// Create the directory `path_` and any missing parents; returns whether this succeeded.
  bool create_folder(const std::string &path_);

  /// A duration in seconds as e.g. "95s", "12min5s" or "2h3min4s".
  std::string time_format(size_t time_in_seconds);

  /// time_format() for other number types; the value is truncated to whole seconds.
  template <typename T>
    requires(!std::is_same_v<T, size_t>)
  std::string time_format(T time_in_seconds)
  {
    return time_format(static_cast<size_t>(time_in_seconds));
  }

  /// A duration in milliseconds as e.g. "250ms", "3s250ms", "12min5s" or "2h3min" (seconds dropped above one hour).
  std::string time_format_ms(size_t time_in_milliseconds);

  /// Whether `str` ends with `suffix`.
  bool has_suffix(const std::string &str, const std::string &suffix);
} // namespace DiFfRG
