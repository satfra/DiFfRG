#pragma once

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/physics/interpolation.hh>

#include <autodiff/forward/real/real.hpp>

#include <hdf5lib/hdf5.hh>

namespace DiFfRG::hdf5
{
  template <typename T> struct TypeTrait<DiFfRG::complex<T>> {
    static Datatype get()
    {
      auto t = Datatype::compound(2 * sizeof(T));
      t.insert("real", 0, type_of<T>());
      t.insert("imag", sizeof(T), type_of<T>());
      return t;
    }
  };

  template <std::size_t N, typename T> struct TypeTrait<autodiff::Real<N, T>> {
    static Datatype get()
    {
      auto t = Datatype::compound((N + 1) * sizeof(T));
      t.insert("val", 0, type_of<T>());
      for (std::size_t i = 1; i <= N; ++i)
        t.insert("d_" + std::to_string(i), i * sizeof(T), type_of<T>());
      return t;
    }
  };

  template <typename T, std::size_t N> struct TypeTrait<std::array<T, N>> {
    static Datatype get()
    {
      auto t = Datatype::compound(N * sizeof(T));
      for (std::size_t i = 0; i < N; ++i)
        t.insert("component " + std::to_string(i), i * sizeof(T), type_of<T>());
      return t;
    }
  };

  template <typename T, std::size_t N>
    requires(!std::is_same_v<std::array<T, N>, DiFfRG::device::array<T, N>>)
  struct TypeTrait<DiFfRG::device::array<T, N>> {
    static Datatype get()
    {
      auto t = Datatype::compound(N * sizeof(T));
      for (std::size_t i = 0; i < N; ++i)
        t.insert("component " + std::to_string(i), i * sizeof(T), type_of<T>());
      return t;
    }
  };
} // namespace DiFfRG::hdf5
