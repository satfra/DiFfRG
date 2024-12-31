#pragma once

// standard library
#include <future>
#include <memory>
#include <tbb/tbb.h>
#include <tuple>
#include <vector>

// DiFfRG
#include <DiFfRG/discretization/grid/coordinates.hh>

namespace DiFfRG
{
  /**
   * @brief For a given set of coordinates and arguments, this function will call fun.request(k, coordinates...,
   * args...) for each coordinate in coordinates.
   *
   * @tparam NT Type of the data requested
   * @tparam FUN Type of the function object
   * @tparam Coordinates Type of the coordinates
   * @tparam T Types of the arguments
   *
   * @param FUN& fun The function object, usually an integrator class.
   * @param Coordinates& coordinates The coordinates object
   * @param const double k current RG scale
   * @param std::tuple<T...> args The arguments to pass to the request function
   *
   * @return std::shared_ptr<std::vector<std::future<NT>>> A vector of futures which will yield the requested data
   */
  template <typename NT, typename FUN, typename Coordinates, typename... T>
  std::shared_ptr<std::vector<std::future<NT>>> request_data(FUN &fun, Coordinates &coordinates, const double k,
                                                             std::tuple<T...> args)
  {
    return std::apply([&](auto... args) { return request_data<NT>(fun, coordinates, k, args...); }, args);
  }

  /**
   * @brief For a given set of coordinates and arguments, this function will call fun.request(k, coordinates...,
   * args...) for each coordinate in coordinates.
   *
   * @tparam NT Type of the data requested
   * @tparam FUN Type of the function object
   * @tparam Coordinates Type of the coordinates
   * @tparam T Types of the arguments
   *
   * @param FUN& fun The function object, usually an integrator class.
   * @param Coordinates& coordinates The coordinates object
   * @param const double k current RG scale
   * @param T... args The arguments to pass to the request function
   *
   * @return std::shared_ptr<std::vector<std::future<NT>>> A vector of futures which will yield the requested data
   */
  template <typename NT, typename FUN, typename Coordinates, typename... T>
  std::shared_ptr<std::vector<std::future<NT>>> request_data(FUN &fun, Coordinates &coordinates, const double k,
                                                             T... args)
  {
    auto futures = std::make_shared<std::vector<std::future<NT>>>();
    auto grid = make_grid(coordinates);

    auto req_point = [&](auto... p) { return fun.template request<NT>(k, p..., args...); };

    for (uint i = 0; i < coordinates.size(); ++i) {
      auto p = grid[i];
      std::apply([&](auto... p) { futures->emplace_back(std::move(req_point(p...))); }, p);
    }

    return futures;
  }

  /**
   * @brief For a given grid and arguments, this function will call fun.request(k, grid[...]..., args...) for each
   * gridpoint in grid.
   *
   * @tparam NT Type of the data requested
   * @tparam FUN Type of the function object
   * @tparam GRID Type of the grid
   * @tparam T Types of the arguments
   *
   * @param FUN& fun The function object, usually an integrator class.
   * @param GRID& grid The grid object
   * @param const double k current RG scale
   * @param std::tuple<T...> args The arguments to pass to the request function
   *
   * @return std::shared_ptr<std::vector<std::future<NT>>> A vector of futures which will yield the requested data
   */
  template <typename NT, typename FUN, typename GRID, typename... T>
  std::shared_ptr<std::vector<std::future<NT>>> request_data(FUN &fun, const GRID &grid, const double k,
                                                             std::tuple<T...> args)
  {
    return std::apply([&](auto... args) { return request_data<NT>(fun, grid, k, args...); }, args);
  }

  /**
   * @brief For a given grid and arguments, this function will call fun.request(k, grid[...]..., args...) for each
   * gridpoint in grid.
   *
   * @tparam NT Type of the data requested
   * @tparam FUN Type of the function object
   * @tparam GRID Type of the grid
   * @tparam T Types of the arguments
   *
   * @param FUN& fun The function object, usually an integrator class.
   * @param GRID& grid The grid object
   * @param const double k current RG scale
   * @param T... args The arguments to pass to the request function
   *
   * @return std::shared_ptr<std::vector<std::future<NT>>> A vector of futures which will yield the requested data
   */
  template <typename NT, typename FUN, typename GRID, typename... T>
  std::shared_ptr<std::vector<std::future<NT>>> request_data(FUN &fun, const GRID &grid, const double k, T... args)
  {
    auto futures = std::make_shared<std::vector<std::future<NT>>>();

    auto req_point = [&](auto... p) { return fun.template request<NT>(k, p..., args...); };

    for (uint i = 0; i < grid.size(); ++i) {
      auto p = grid[i];
      if constexpr (std::is_same_v<decltype(p), double> || std::is_same_v<decltype(p), float>)
        futures->emplace_back(std::move(req_point(p)));
      else
        std::apply([&](auto... p) { futures->emplace_back(std::move(req_point(p...))); }, p);
    }

    return futures;
  }

  /**
   * @brief Obtain data from a vector of futures and store it in a destination array.
   *
   * @tparam NT1 Type of the data in the futures
   * @tparam NT2 Type of the data in the destination array
   * @tparam T Types of the arguments
   *
   * @param futures The vector of futures
   * @param destination The destination array
   */
  template <typename NT1, typename NT2, typename... T>
  void update_data(std::shared_ptr<std::vector<std::future<NT1>>> futures, NT2 *destination)
  {
    for (uint i = 0; i < futures->size(); ++i)
      destination[i] = static_cast<NT2>((*futures)[i].get());
  }

  /**
   * @brief Obtain data from a vector of futures and store it in an interpolator.
   *
   * @tparam NT Type of the data in the futures
   * @tparam INT Type of the interpolator. Must have a method update().
   * @tparam T Types of the arguments
   *
   * @param futures The vector of futures
   * @param destination The interpolator
   */
  template <typename NT, typename INT, typename... T>
  void update_interpolator(std::shared_ptr<std::vector<std::future<NT>>> futures, INT &destination)
  {
    for (uint i = 0; i < futures->size(); ++i)
      destination[i] = (*futures)[i].get();
    destination.update();
  }

  /**
   * @brief For a given grid and arguments, this function will call fun.request(k, grid[...]..., args...) for each
   * gridpoint in grid.
   *
   * @tparam NT Type of the data requested
   * @tparam FUN Type of the function object
   * @tparam GRID Type of the grid
   * @tparam T Types of the arguments
   *
   * @param FUN& fun The function object, usually an integrator class.
   * @param GRID& grid The grid object
   * @param const double k current RG scale
   * @param std::tuple<T...> args The arguments to pass to the request function
   *
   * @return std::shared_ptr<std::vector<std::future<NT>>> A vector of futures which will yield the requested data
   */
  template <typename NT, typename FUN, typename GRID, typename... T>
  void get_data(NT *dest, FUN &fun, const GRID &grid, const double k, std::tuple<T...> args)
  {
    std::apply([&](auto... args) { get_data<NT>(dest, fun, grid, k, args...); }, args);
  }

  /**
   * @brief For a given grid and arguments, this function will call fun.request(k, grid[...]..., args...) for each
   * gridpoint in grid.
   *
   * @tparam NT Type of the data requested
   * @tparam FUN Type of the function object
   * @tparam GRID Type of the grid
   * @tparam T Types of the arguments
   *
   * @param FUN& fun The function object, usually an integrator class.
   * @param GRID& grid The grid object
   * @param const double k current RG scale
   * @param T... args The arguments to pass to the request function
   *
   * @return std::shared_ptr<std::vector<std::future<NT>>> A vector of futures which will yield the requested data
   */
  template <typename NT, typename FUN, typename GRID, typename... T>
  void get_data(NT *dest, FUN &fun, const GRID &grid, const double k, T... args)
  {
    auto req_point = [&](auto... p) { return fun.template get<NT>(k, p..., args...); };

    // for (uint i = 0; i < grid.size(); ++i) {
    tbb::parallel_for(tbb::blocked_range<uint>(0, grid.size()), [&](const tbb::blocked_range<uint> &r) {
      for (uint i = r.begin(); i < r.end(); ++i) {
        auto p = grid[i];
        if constexpr (std::is_same_v<decltype(p), double> || std::is_same_v<decltype(p), float>)
          dest[i] = req_point(p);
        else
          std::apply([&](auto... p) { dest[i] = req_point(p...); }, p);
      }
    });
  }

} // namespace DiFfRG
