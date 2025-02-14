#pragma once

// standard library
#include <cmath>
#include <cstring>
#include <string_view>
#include <tuple>
#include <utility>

// DiFfRG
#include <DiFfRG/common/fixed_string.hh>

namespace DiFfRG
{
  /**
   * @brief Check if two strings are equal at compile time.
   */
  constexpr bool strings_equal(char const *a, char const *b) { return std::string_view(a) == b; }

  /**
   * @brief A class to store a tuple with elements that can be accessed by name.
   * The names are stored as FixedString objects and their lookup is done at compile time.
   *
   * @tparam tuple_type The type of the underlying tuple.
   * @tparam strs The names of the elements in the tuple.
   */
  template <typename tuple_type, FixedString... strs> struct named_tuple {
    static_assert(sizeof...(strs) == std::tuple_size_v<tuple_type>,
                  "Number of names must match number of elements in tuple");
    tuple_type tuple;

    constexpr operator tuple_type &() { return tuple; }

    static constexpr size_t size = sizeof...(strs);
    static constexpr std::array<const char *, size> names{{strs...}};

    // If two names are the same, the program should not compile
    static_assert(
        []<size_t... I>(std::index_sequence<I...>) {
          for (size_t i : {I...})
            for (size_t j : {I...})
              if (i != j && strings_equal(names[i], names[j])) return false;
          return true;
        }(std::make_index_sequence<size>{}),
        "Names of a named_tuple must be unique!");

    named_tuple(tuple_type &&t) : tuple(t) {}
    named_tuple(tuple_type &t) : tuple(t) {}

    template <typename... T> static constexpr auto as(std::tuple<T...> &&tup)
    {
      return named_tuple<std::tuple<T...>, strs...>(tup);
    }
    template <typename... T> static constexpr auto as(std::tuple<T...> &tup)
    {
      return named_tuple<std::tuple<T...>, strs...>(tup);
    }

    static consteval size_t get_idx(const char *name)
    {
      size_t running_sum = 0;
      for (size_t i = 0; i < names.size(); ++i) {
        // this is a way to compare two strings at compile time https://stackoverflow.com/a/53762940
        if (strings_equal(names[i], name)) return i;
        running_sum += 1;
      }
      // produce a compile-time error if the name is not found in the list
      return size != running_sum ? 0
                                 : throw std::invalid_argument(
                                       "named_tuple::get_idx: Name \"" + std::string(name) +
                                       "\" not found. Available names are: " + ((std::string(strs) + "; ") + ...));
    }

    template <size_t idx> auto &get() { return std::get<idx>(tuple); }
    template <size_t idx> const auto &get() const { return std::get<idx>(tuple); }
  };

  /**
   * @brief get a reference to the element with the given name
   */
  template <FixedString name, typename tuple_type, FixedString... strs>
  constexpr auto &get(named_tuple<tuple_type, strs...> &ob)
  {
    constexpr size_t idx = named_tuple<tuple_type, strs...>::get_idx(name);
    return ob.template get<idx>();
  }
  template <FixedString name, typename tuple_type, FixedString... strs>
  constexpr auto &get(named_tuple<tuple_type, strs...> &&ob)
  {
    constexpr size_t idx = named_tuple<tuple_type, strs...>::get_idx(name);
    return ob.template get<idx>();
  }
  template <FixedString name, typename tuple_type, FixedString... strs>
  constexpr auto &get(const named_tuple<tuple_type, strs...> &ob)
  {
    return std::get<named_tuple<tuple_type, strs...>::get_idx(name)>(ob.tuple);
  }

  template <size_t idx, typename tuple_type, FixedString... strs>
  constexpr auto &get(named_tuple<tuple_type, strs...> &ob)
  {
    return ob.template get<idx>();
  }
  template <size_t idx, typename tuple_type, FixedString... strs>
  constexpr auto &get(named_tuple<tuple_type, strs...> &&ob)
  {
    return ob.template get<idx>();
  }
  template <size_t idx, typename tuple_type, FixedString... strs>
  constexpr auto &get(const named_tuple<tuple_type, strs...> &ob)
  {
    return ob.template get<idx>();
  }
} // namespace DiFfRG

namespace std
{
  template <size_t idx, typename tuple_type, DiFfRG::FixedString... strs>
  constexpr auto &get(DiFfRG::named_tuple<tuple_type, strs...> &ob)
  {
    return ob.template get<idx>();
  }
  template <size_t idx, typename tuple_type, DiFfRG::FixedString... strs>
  constexpr auto &get(DiFfRG::named_tuple<tuple_type, strs...> &&ob)
  {
    return ob.template get<idx>();
  }
  template <size_t idx, typename tuple_type, DiFfRG::FixedString... strs>
  constexpr auto &get(const DiFfRG::named_tuple<tuple_type, strs...> &ob)
  {
    return ob.template get<idx>();
  }

  // tuple_size_v
  template <typename tuple_type, DiFfRG::FixedString... strs>
  struct tuple_size<DiFfRG::named_tuple<tuple_type, strs...>> : std::integral_constant<size_t, sizeof...(strs)> {
  };
} // namespace std

// Intense Voodoo
namespace DiFfRG
{
  // ----------------------------------------------------------------------
  // AD helpers. The following are used to process the jacobians via
  // automatic differentiation. Most of these methods are used in the
  // file model/AD.hh
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

    /**
     * @brief Check whether the matrix contains only finite values.
     */
    bool is_finite() const
    {
      if constexpr (std::is_floating_point_v<NT> || std::is_same_v<NT, autodiff::real>) {
        for (uint i = 0; i < N * M; ++i)
          if (!isfinite(data[i])) return false;
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
  template <uint n, typename NT, typename Vector> std::array<NT, n> vector_to_array(const Vector &v)
  {
    std::array<NT, n> x;
    for (uint i = 0; i < n; ++i)
      x[i] = v[i];
    return x;
  }

  template <typename T, std::size_t... Indices>
  auto vectorToTupleHelper(const std::vector<T> &v, std::index_sequence<Indices...>)
  {
    return std::tie(v[Indices]...);
  }
  template <std::size_t N, typename T> auto vector_to_tuple(const std::vector<T> &v)
  {
    assert(v.size() >= N);
    return vectorToTupleHelper(v, std::make_index_sequence<N>());
  }

  template <typename Head, typename... Tail> constexpr auto tuple_tail(const std::tuple<Head, Tail...> &t)
  {
    return std::apply([](auto & /*head*/, auto &...tail) { return std::tie(tail...); }, t);
  }
  // also for named tuple
  template <typename tuple_type, FixedString... strs>
  constexpr auto tuple_tail(const named_tuple<tuple_type, strs...> &t)
  {
    return tuple_tail(t.tuple);
  }

  template <int i, typename Head, typename... Tail> constexpr auto tuple_last(const std::tuple<Head, Tail...> &t)
  {
    if constexpr (sizeof...(Tail) == i)
      return std::apply([](auto & /*head*/, auto &...tail) { return std::tie(tail...); }, t);
    else
      return std::apply([](auto & /*head*/, auto &...tail) { return tuple_last<i>(std::tie(tail...)); }, t);
  }
  // also for named tuple
  template <int i, typename tuple_type, FixedString... strs>
  constexpr auto tuple_last(const named_tuple<tuple_type, strs...> &t)
  {
    return tuple_last<i>(t.tuple);
  }

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
  template <int i, typename tuple_type, FixedString... strs>
  constexpr auto tuple_first(const named_tuple<tuple_type, strs...> &t)
  {
    return tuple_first<i>(t.tuple);
  }

  // ----------------------------------------------------------------------
  // Helper functions to get the local solution at a given q_index.
  // This is specifically for the local solutions of the subsystems,
  // i.e. only used by the LDG assembler.
  // ----------------------------------------------------------------------

  template <typename T, size_t N, size_t... IDXs>
  auto _local_sol_tuple(const std::array<T, N> &a, std::index_sequence<IDXs...>, uint q_index)
  {
    return std::tie(a[IDXs][q_index]...);
  }
  template <typename T, size_t N> auto local_sol_q(const std::array<T, N> &a, uint q_index)
  {
    return _local_sol_tuple(a, std::make_index_sequence<N>{}, q_index);
  }

  template <typename T_inner, typename Model, size_t... IDXs> auto _jacobian_tuple(std::index_sequence<IDXs...>)
  {
    return std::tuple{SimpleMatrix<T_inner, Model::Components::count_fe_functions(0),
                                   Model::Components::count_fe_functions(IDXs)>()...};
  }
  template <typename T_inner, typename Model> auto jacobian_tuple()
  {
    return _jacobian_tuple<T_inner, Model>(std::make_index_sequence<Model::Components::count_fe_subsystems()>{});
  }

  template <typename T_inner, typename Model, size_t... IDXs> auto _jacobian_2_tuple(std::index_sequence<IDXs...>)
  {
    return std::tuple{std::array<
        SimpleMatrix<T_inner, Model::Components::count_fe_functions(0), Model::Components::count_fe_functions(IDXs)>,
        2>()...};
  }
  template <typename T_inner, typename Model> auto jacobian_2_tuple()
  {
    return _jacobian_2_tuple<T_inner, Model>(std::make_index_sequence<Model::Components::count_fe_subsystems()>{});
  }
} // namespace DiFfRG
