#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>

// external libraries
#include <autodiff/forward/real.hpp>

// std
#include <type_traits>

// NOTE: the linear-algebra type map (get_type::NumberType / SparsityPattern /
// InverseSparseMatrixType / BlockVectorType, is_distributed_la, SupportedVectorType and
// the build-configuration defaults) lives in DiFfRG/common/linear_algebra.hh. It is
// deliberately not included here: everything under physics/ needs only get_type::ctype,
// and pulling deal.II's sparse_direct.h, block_sparse_matrix.h and the PETSc headers into
// every Kokkos/CUDA translation unit through this file cost real compile time.

namespace DiFfRG
{
  template <typename T>
  concept is_container = requires(T x) { x[0]; };

  template <size_t N, typename T>
  concept is_sized_container = requires(T x) {
    x[0];
    requires x.size() == N;
  };

  // we need a concept to check if a type has an operator() with N arguments of type ValueType
  template <typename T, typename ValueType, size_t N> struct has_n_call_operator_helper {
    template <size_t... Is> static constexpr bool tryit(std::integer_sequence<size_t, Is...>)
    {
      if constexpr (requires(T t, ValueType v) { t.operator()((ValueType{} * Is)...); }) {
        return true;
      } else {
        return false;
      }
    }

    static constexpr bool value = tryit(std::make_index_sequence<N>{});
  };

  template <typename T, typename ValueType, size_t N>
  concept has_n_call_operator = has_n_call_operator_helper<T, ValueType, N>::value;

  namespace get_type
  {
    namespace internal
    {
      template <typename CT> struct _ctype;

      template <> struct _ctype<float> {
        using value = float;
      };

      template <> struct _ctype<double> {
        using value = double;
      };

      template <> struct _ctype<complex<float>> {
        using value = float;
      };

      template <> struct _ctype<complex<double>> {
        using value = double;
      };

      template <size_t N, typename T> struct _ctype<autodiff::Real<N, T>> : _ctype<T> {
      };

      template <typename T> inline constexpr bool _is_autodiff = false;
      template <size_t N, typename U> inline constexpr bool _is_autodiff<autodiff::Real<N, U>> = true;
    } // namespace internal

    template <typename CT> using ctype = typename internal::_ctype<CT>::value;

    namespace internal
    {
      template <typename T> struct _double_precision {
        using value = T;
      };

      template <> struct _double_precision<float> {
        using value = double;
      };

      template <> struct _double_precision<complex<float>> {
        using value = complex<double>;
      };

      template <size_t N> struct _double_precision<autodiff::Real<N, float>> {
        using value = autodiff::Real<N, double>;
      };
    } // namespace internal

    /// The double-precision counterpart of a single-precision type (float -> double,
    /// complex<float> -> complex<double>, Real<N, float> -> Real<N, double>); every other type maps
    /// to itself.
    template <typename T> using double_precision = typename internal::_double_precision<T>::value;

    namespace internal
    {
      template <typename T> struct _single_precision {
        using value = T;
      };

      template <> struct _single_precision<double> {
        using value = float;
      };

      template <> struct _single_precision<complex<double>> {
        using value = complex<float>;
      };
    } // namespace internal

    /// The single-precision counterpart of a double-precision type (double -> float,
    /// complex<double> -> complex<float>); every other type maps to itself.
    template <typename T> using single_precision = typename internal::_single_precision<T>::value;

    template <typename T> inline constexpr bool is_autodiff = internal::_is_autodiff<T>;
  } // namespace get_type
} // namespace DiFfRG
