#pragma once

// standard library
#include <cmath>
#include <cstring>
#include <iostream>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

// DiFfRG
#include <DiFfRG/common/fixed_string.hh>

/**
 * @file tuples.hh
 * @brief Tuples whose elements are accessed by name, as used for the arguments of model functions.
 *
 * A model's flux, source, `dt_variables` etc. receive their arguments as a named_tuple and read them with
 * `get<"name">(data)`:
 * @code
 * template <typename Vector, typename Solution> void dt_variables(Vector &residual, const Solution &data) const
 * {
 *   const auto &variables = get<"variables">(data);
 *   residual[0] = variables[0];
 * }
 * @endcode
 * Which names exist depends on the function and the discretization; tuple_has checks for one at compile time.
 */

namespace DiFfRG
{
  namespace internal
  {
    /// Compile-time string comparison.
    constexpr bool strings_equal(char const *a, char const *b) { return std::string_view(a) == b; }
  } // namespace internal

  /// The list of element names of a named_tuple; names must be unique.
  template <FixedString... strs> struct StringSet {
    static constexpr size_t size = sizeof...(strs);
    static constexpr std::array<const char *, size> names{{strs...}};

    // If two names are the same, the program should not compile
    static_assert(
        []<size_t... I>(std::index_sequence<I...>) {
          for (size_t i : {I...})
            for (size_t j : {I...})
              if (i != j && internal::strings_equal(names[i], names[j])) return false;
          return true;
        }(std::make_index_sequence<size>{}),
        "Names of a StringSet must be unique!");
  };

  /**
   * @brief A tuple whose elements can be accessed by name, `get<"name">(t)`, or by index, `get<i>(t)`.
   *
   * The name lookup happens at compile time; an unknown name is a compile error.
   *
   * @tparam tuple_type the underlying `std::tuple`, usually of references.
   * @tparam tuple_names a StringSet with one name per element.
   */
  template <typename tuple_type, typename tuple_names> struct named_tuple {
    static_assert(tuple_names::size == std::tuple_size_v<tuple_type>,
                  "Number of names must match number of elements in tuple");
    tuple_type tuple;

    constexpr operator tuple_type &() { return tuple; }

    static constexpr size_t size = tuple_names::size;
    static constexpr auto names = tuple_names::names;

    // If two names are the same, the program should not compile
    static_assert(
        []<size_t... I>(std::index_sequence<I...>) {
          for (size_t i : {I...})
            for (size_t j : {I...})
              if (i != j && internal::strings_equal(names[i], names[j])) return false;
          return true;
        }(std::make_index_sequence<size>{}),
        "Names of a named_tuple must be unique!");

    named_tuple(tuple_type &&t) : tuple(t) {}
    named_tuple(tuple_type &t) : tuple(t) {}

    /// A named_tuple with the same names around another tuple, e.g. one holding autodiff copies of the elements.
    template <typename... T> static constexpr auto as(std::tuple<T...> &&tup)
    {
      return named_tuple<std::tuple<T...>, tuple_names>(tup);
    }

    /// A named_tuple with the same names around another tuple, e.g. one holding autodiff copies of the elements.
    template <typename... T> static constexpr auto as(std::tuple<T...> &tup)
    {
      return named_tuple<std::tuple<T...>, tuple_names>(tup);
    }

    /// Index of the element called `name`; a compile error if there is none.
    static consteval size_t get_idx(const char *name)
    {
      size_t running_sum = 0;
      for (size_t i = 0; i < names.size(); ++i) {
        // this is a way to compare two strings at compile time https://stackoverflow.com/a/53762940
        if (internal::strings_equal(names[i], name)) return i;
        running_sum += 1;
      }
      // produce a compile-time error if the name is not found in the list
      return size != running_sum
                 ? 0
                 : throw std::invalid_argument("named_tuple::get_idx: Name \"" + std::string(name) +
                                               "\" not found. Available names are: " + ((std::string("") + "; ") + ""));
    }

    /// Whether an element has the given name.
    static consteval bool contains(const char *name)
    {
      for (size_t i = 0; i < names.size(); ++i)
        if (internal::strings_equal(names[i], name)) return true;
      return false;
    }

    /// Element number `idx`.
    template <size_t idx> auto &get() { return std::get<idx>(tuple); }
    template <size_t idx> const auto &get() const { return std::get<idx>(tuple); }
  };

  /**
   * @brief Whether the named_tuple type @p NamedTuple has an element called @p name, e.g.
   * `if constexpr (tuple_has<"fe_derivatives", Solution>)` in a model whose flux serves several assemblers.
   */
  template <FixedString name, typename NamedTuple>
  constexpr bool tuple_has = std::remove_cvref_t<NamedTuple>::contains(name);

  /**
   * @brief The element called `name` of a named_tuple, e.g. `get<"variables">(data)`.
   *
   * For a temporary named_tuple the returned reference is only valid if the tuple holds references.
   */
  template <FixedString name, typename tuple_type, typename strSet>
  constexpr auto &get(named_tuple<tuple_type, strSet> &ob)
  {
    constexpr size_t idx = named_tuple<tuple_type, strSet>::get_idx(name);
    return ob.template get<idx>();
  }
  template <FixedString name, typename tuple_type, typename strSet>
  constexpr auto &get(named_tuple<tuple_type, strSet> &&ob)
  {
    constexpr size_t idx = named_tuple<tuple_type, strSet>::get_idx(name);
    return ob.template get<idx>();
  }
  template <FixedString name, typename tuple_type, typename strSet>
  constexpr auto &get(const named_tuple<tuple_type, strSet> &ob)
  {
    return std::get<named_tuple<tuple_type, strSet>::get_idx(name)>(ob.tuple);
  }

  /// Element number `idx` of a named_tuple.
  template <size_t idx, typename tuple_type, typename strSet> constexpr auto &get(named_tuple<tuple_type, strSet> &ob)
  {
    return ob.template get<idx>();
  }
  template <size_t idx, typename tuple_type, typename strSet> constexpr auto &get(named_tuple<tuple_type, strSet> &&ob)
  {
    return ob.template get<idx>();
  }
  template <size_t idx, typename tuple_type, typename strSet>
  constexpr auto &get(const named_tuple<tuple_type, strSet> &ob)
  {
    return ob.template get<idx>();
  }
} // namespace DiFfRG

// Intense Voodoo
namespace DiFfRG
{
  // ----------------------------------------------------------------------
  // AD helpers. The following are used to process the jacobians via
  // automatic differentiation. Most of these methods are used in the
  // file model/ad.hh
  // ----------------------------------------------------------------------

  /**
   * @brief A simple NxM-matrix class, which is used for cell-wise Jacobians.
   */
  template <typename NT, uint N, uint M = N> class SimpleMatrix
  {
  public:
    SimpleMatrix() : data() {}

    /**
     * @brief Access the matrix entry at (n,m).
     */
    NT &operator()(const uint n, const uint m) { return data[n * M + m]; }

    /**
     * @brief Access the matrix entry at (n,m).
     */
    const NT &operator()(const uint n, const uint m) const { return data[n * M + m]; }

    /**
     * @brief Set all entries to zero.
     */
    void clear()
    {
      for (uint i = 0; i < N * M; ++i)
        data[i] = 0.;
    }

    /// Whether all entries are finite; always true for non-floating-point `NT`.
    bool is_finite() const
    {
      if constexpr (std::is_floating_point_v<NT>) {
        for (uint i = 0; i < N * M; ++i)
          if (!std::isfinite(data[i])) return false;
      }
      return true;
    }

    /**
     * @brief Print the matrix to the console.
     */
    void print() const
    {
      for (uint i = 0; i < N; ++i) {
        for (uint j = 0; j < M; ++j)
          std::cout << data[i * M + j] << " ";
        std::cout << std::endl;
      }
    }

  private:
    std::array<NT, N * M> data;
  };

  /// The first `n` entries of `v` as a `std::array<NT, n>`.
  template <uint n, typename NT, typename Vector> std::array<NT, n> vector_to_array(const Vector &v)
  {
    std::array<NT, n> x;
    for (uint i = 0; i < n; ++i)
      x[i] = v[i];
    return x;
  }

  namespace internal
  {
    template <typename T, std::size_t... Indices>
    auto vector_to_tuple_helper(const std::vector<T> &v, std::index_sequence<Indices...>)
    {
      return std::tie(v[Indices]...);
    }
  } // namespace internal
  /// A tuple of references to the first `N` entries of `v`.
  template <std::size_t N, typename T> auto vector_to_tuple(const std::vector<T> &v)
  {
    assert(v.size() >= N);
    return internal::vector_to_tuple_helper(v, std::make_index_sequence<N>());
  }

  /// A tuple of references to all elements of `t` but the first.
  template <typename Head, typename... Tail> constexpr auto tuple_tail(const std::tuple<Head, Tail...> &t)
  {
    return std::apply([](auto & /*head*/, auto &...tail) { return std::tie(tail...); }, t);
  }
  // also for named tuple
  template <typename tuple_type, typename strSet> constexpr auto tuple_tail(const named_tuple<tuple_type, strSet> &t)
  {
    return tuple_tail(t.tuple);
  }

  /// A tuple of references to the last `i` elements of `t`.
  template <int i, typename Head, typename... Tail> constexpr auto tuple_last(const std::tuple<Head, Tail...> &t)
  {
    if constexpr (sizeof...(Tail) == i)
      return std::apply([](auto & /*head*/, auto &...tail) { return std::tie(tail...); }, t);
    else
      return std::apply([](auto & /*head*/, auto &...tail) { return tuple_last<i>(std::tie(tail...)); }, t);
  }
  // also for named tuple
  template <int i, typename tuple_type, typename strSet>
  constexpr auto tuple_last(const named_tuple<tuple_type, strSet> &t)
  {
    return tuple_last<i>(t.tuple);
  }

  /// A tuple of references to the first `i` elements of `t`.
  template <int i, typename Head, typename... Tail> constexpr auto tuple_first(const std::tuple<Head, Tail...> &t)
  {
    if constexpr (i == 0)
      return std::tuple();
    else if constexpr (i == 1)
      return std::apply([](auto &head, auto &.../*tail*/) { return std::tie(head); }, t);
    else
      return std::apply(
          [](auto &head, auto &...tail) {
            return std::tuple_cat(std::tie(head), tuple_first<i - 1>(std::tie(tail...)));
          },
          t);
  }
  // also for named tuple
  template <int i, typename tuple_type, typename strSet>
  constexpr auto tuple_first(const named_tuple<tuple_type, strSet> &t)
  {
    return tuple_first<i>(t.tuple);
  }

  // ----------------------------------------------------------------------
  // Helper functions to get the local solution at a given q_index.
  // This is specifically for the local solutions of the subsystems,
  // i.e. only used by the LDG assembler.
  // ----------------------------------------------------------------------

  namespace internal
  {
    template <typename T, size_t N, size_t... IDXs>
    auto local_sol_tuple(const std::array<T, N> &a, std::index_sequence<IDXs...>, uint q_index)
    {
      return std::tie(a[IDXs][q_index]...);
    }

    template <typename T_inner, typename Model, size_t... IDXs> auto jacobian_tuple(std::index_sequence<IDXs...>)
    {
      return std::tuple{SimpleMatrix<T_inner, Model::Components::count_fe_functions(0),
                                     Model::Components::count_fe_functions(IDXs)>()...};
    }

    template <typename T_inner, typename Model, size_t... IDXs> auto jacobian_2_tuple(std::index_sequence<IDXs...>)
    {
      return std::tuple{std::array<
          SimpleMatrix<T_inner, Model::Components::count_fe_functions(0), Model::Components::count_fe_functions(IDXs)>,
          2>()...};
    }
  } // namespace internal

  template <typename T, size_t N> auto local_sol_q(const std::array<T, N> &a, uint q_index)
  {
    return internal::local_sol_tuple(a, std::make_index_sequence<N>{}, q_index);
  }

  template <typename T_inner, typename Model> auto jacobian_tuple()
  {
    return internal::jacobian_tuple<T_inner, Model>(
        std::make_index_sequence<Model::Components::count_fe_subsystems()>{});
  }

  template <typename T_inner, typename Model> auto jacobian_2_tuple()
  {
    return internal::jacobian_2_tuple<T_inner, Model>(
        std::make_index_sequence<Model::Components::count_fe_subsystems()>{});
  }
} // namespace DiFfRG
