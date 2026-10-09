#pragma once

#include <initializer_list>

namespace DiFfRG
{
  /**
   * @brief A fixed size compile-time string.
   *
   * @tparam N The size of the string.
   */
  template <unsigned N> struct FixedString {
    char buf[N + 1]{};

    /**
     * @brief Construct a new Fixed String object
     *
     * @param s The string to be copied.
     * @note The string must be of size N.
     */
    constexpr FixedString(char const *s)
    {
      for (unsigned i = 0; i != N; ++i)
        buf[i] = s[i];
    }

    constexpr FixedString(std::initializer_list<char> s)
    {
      unsigned i = 0;
      for (auto c : s) {
        buf[i] = c;
        ++i;
      }
    }

    constexpr FixedString(char const (&arr)[N])
    {
      for (unsigned i = 0; i < N; ++i)
        buf[i] = arr[i];
    }

    constexpr operator char const *() const { return buf; }

    auto operator<=>(const FixedString &) const = default;
  };

  template <unsigned N> FixedString(char const (&)[N]) -> FixedString<N - 1>;

  template <unsigned N1, unsigned N2> consteval bool strings_equal(FixedString<N1> s1, FixedString<N2> s2)
  {
    if (N1 != N2) return false;
    for (unsigned i = 0; i < N1; ++i)
      if (s1.buf[i] != s2.buf[i]) return false;
    return true;
  }
} // namespace DiFfRG
