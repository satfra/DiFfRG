#pragma once

#include <initializer_list>

namespace DiFfRG
{
  /**
   * @brief A string usable as a template argument, e.g. the names in `get<"variables">(data)`.
   *
   * @tparam N the length of the string, not counting the terminating null character.
   */
  template <unsigned N> struct FixedString {
    char buf[N + 1]{};

    /// Copy the first \f$N\f$ characters of `s`.
    constexpr FixedString(char const *s)
    {
      for (unsigned i = 0; i != N; ++i)
        buf[i] = s[i];
    }

    /// From a list of at most \f$N\f$ characters.
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

    /// The null-terminated string.
    constexpr operator char const *() const { return buf; }

    auto operator<=>(const FixedString &) const = default;
  };

  template <unsigned N> FixedString(char const (&)[N]) -> FixedString<N - 1>;

  namespace internal
  {
    /// Compile-time string comparison.
    template <unsigned N1, unsigned N2> consteval bool strings_equal(FixedString<N1> s1, FixedString<N2> s2)
    {
      if (N1 != N2) return false;
      for (unsigned i = 0; i < N1; ++i)
        if (s1.buf[i] != s2.buf[i]) return false;
      return true;
    }
  } // namespace internal
} // namespace DiFfRG
