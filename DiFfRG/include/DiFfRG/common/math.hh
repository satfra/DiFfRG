#pragma once

// DiFfRG
#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/utils.hh>

// standard library
#include <cmath>
#include <type_traits>

// external libraries
#include <Eigen/Dense>
#include <autodiff/forward/real.hpp>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

namespace DiFfRG
{
  /**
   * @brief Finite-ness check for autodiff::real
   *
   * @param x Number to check
   * @return Whether x and its derivative are finite
   */
  template <size_t N, typename T> bool isfinite(const autodiff::Real<N, T> &x)
  {
    return std::isfinite(autodiff::val(x)) && std::isfinite(autodiff::derivative(x));
  }
  using std::isfinite;

  /**
   * @brief A compile-time evaluatable power function for whole number exponents
   *
   * @tparam n Exponent of type int
   * @tparam RF Type of argument
   * @param x Argument
   * @return x^n
   */
  template <int n, typename NumberType>
    requires requires(NumberType x) {
      x *x;
      NumberType(1.) / x;
    }
  constexpr __forceinline__ __host__ __device__ NumberType powr(const NumberType x)
  {
    if constexpr (n == 0)
      return NumberType(1.);
    else if constexpr (n < 0)
      return NumberType(1.) / powr<-n, NumberType>(x);
    else if constexpr (n > 1)
      return x * powr<n - 1, NumberType>(x);
    else
      return x;
  }

  /**
   * @brief Volume of a d-dimensional sphere
   *
   * @tparam NT Type of the number
   * @param d Dimension of the sphere
   */
  template <typename NT> constexpr __forceinline__ __host__ __device__ double V_d(NT d)
  {
    using std::pow;
    using std::tgamma;
    return pow(M_PI, d / 2.) / tgamma(d / 2. + 1.);
  }

  /**
   * @brief Volume of a d-dimensional sphere with extent
   *
   * @tparam NT1 Type of the number
   * @tparam NT2 Type of the extent
   * @param d Dimension of the sphere
   * @param extent Extent of the sphere
   */
  template <typename NT1, typename NT2> constexpr __forceinline__ __host__ __device__ double V_d(NT1 d, NT2 extent)
  {
    using std::pow;
    using std::tgamma;
    return pow(M_PI, d / 2.) / tgamma(d / 2. + 1.) * pow(extent, d);
  }

  /**
   * @brief Surface of a d-dimensional sphere
   *
   * @tparam NT Type of the number
   * @param d Dimension of the sphere
   */
  template <typename NT> constexpr __forceinline__ __host__ __device__ double S_d(NT d)
  {
    using std::pow;
    using std::tgamma;
    return 2. * pow(M_PI, d / 2.) / tgamma(d / 2.);
  }

  /**
   * @brief A compile-time evaluatable theta function
   */
  template <typename NumberType>
    requires requires(NumberType x) { x >= 0; }
  constexpr __forceinline__ __host__ __device__ auto heaviside_theta(const NumberType x)
  {
    if constexpr (std::is_same_v<NumberType, autodiff::real>)
      return x >= 0. ? 1. : 0.;
    else
      return x >= static_cast<NumberType>(0) ? static_cast<NumberType>(1) : static_cast<NumberType>(0);
  }

  /**
   * @brief A compile-time evaluatable sign function
   */
  template <typename NumberType>
    requires requires(NumberType x) { x >= 0; }
  constexpr __forceinline__ __host__ __device__ auto sign(const NumberType x)
  {
    if constexpr (std::is_same_v<NumberType, autodiff::real>)
      return x >= 0. ? 1. : -1.;
    else
      return x >= static_cast<NumberType>(0) ? static_cast<NumberType>(1) : static_cast<NumberType>(-1);
  }

  /**
   * @brief Function to evaluate whether two floats are equal to numerical precision.
   * Tests for both relative and absolute equality.
   *
   * @param eps_ Precision with which to compare a and b
   * @return bool
   */
  template <typename T1, typename T2, typename T3>
    requires(std::is_floating_point<T1>::value || std::is_same_v<T1, autodiff::real> || is_complex<T1>::value) &&
            (std::is_floating_point<T2>::value || std::is_same_v<T2, autodiff::real> || is_complex<T2>::value) &&
            std::is_floating_point<T3>::value
  bool __forceinline__ __host__ __device__ is_close(T1 a, T2 b, T3 eps_)
  {
    if constexpr (is_complex<T1>::value || is_complex<T2>::value) {
      return is_close(real(a), real(b), eps_) && is_close(imag(a), imag(b), eps_);
    } else if constexpr (std::is_same_v<T1, autodiff::real> || std::is_same_v<T2, autodiff::real>)
      return is_close((double)a, (double)b, (double)eps_);
    else {
      T1 diff = std::fabs(a - b);
      if (diff <= eps_) return true;
      if (diff <= std::fmax(std::fabs(a), std::fabs(b)) * eps_) return true;
    }
    return false;
  }

  /**
   * @brief Function to evaluate whether two floats are equal to numerical precision.
   * Tests for both relative and absolute equality.
   *
   * @return bool
   */
  template <typename T1, typename T2>
    requires(std::is_floating_point<T1>::value || std::is_same_v<T1, autodiff::real> || is_complex<T1>::value) &&
            (std::is_floating_point<T2>::value || std::is_same_v<T2, autodiff::real> || is_complex<T1>::value)
  bool __forceinline__ __host__ __device__ is_close(T1 a, T2 b)
  {
    if constexpr (is_complex<T1>::value || is_complex<T2>::value) {
      return is_close(real(a), real(b)) && is_close(imag(a), imag(b));
    } else if constexpr (std::is_same_v<T1, autodiff::real> || std::is_same_v<T2, autodiff::real>) {
      constexpr auto eps_ = std::numeric_limits<double>::epsilon() * 10.;
      return is_close((double)a, (double)b, eps_);
    } else {
      constexpr auto eps_ = std::max(std::numeric_limits<T1>::epsilon(), std::numeric_limits<T2>::epsilon());
      return is_close(a, b, eps_);
    }
    return false;
  }

  /**
   * @brief A dot product which takes the dot product between a1 and a2, assuming each has n entries which can be
   * accessed via the [] operator.
   */
  template <uint n, typename NT, typename A1, typename A2>
    requires requires(A1 a1, A2 a2) { a1[0] * a2[0]; }
  NT dot(const A1 &a1, const A2 &a2)
  {
    NT ret = a1[0] * a2[0];
    for (uint i = 1; i < n; ++i)
      ret += a1[i] * a2[i];
    return ret;
  }

  /**
   * @brief Converts a dealii vector to an Eigen vector
   *
   * @param dealii a dealii vector
   * @param eigen an Eigen vector
   */
  void dealii_to_eigen(const dealii::Vector<double> &dealii, Eigen::VectorXd &eigen);

  /**
   * @brief Converts a dealii block vector to an Eigen vector
   *
   * @param dealii a dealii block vector
   * @param eigen an Eigen vector
   */
  void dealii_to_eigen(const dealii::BlockVector<double> &dealii, Eigen::VectorXd &eigen);

  /**
   * @brief Converts an Eigen vector to a dealii vector
   *
   * @param eigen an Eigen vector
   * @param dealii a dealii vector
   */
  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::Vector<double> &dealii);

  /**
   * @brief Converts an Eigen vector to a dealii block vector
   *
   * @param eigen an Eigen vector
   * @param dealii a dealii block vector
   */
  void eigen_to_dealii(const Eigen::VectorXd &eigen, dealii::BlockVector<double> &dealii);
} // namespace DiFfRG