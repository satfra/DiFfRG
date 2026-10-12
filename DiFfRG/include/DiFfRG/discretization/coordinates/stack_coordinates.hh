#pragma once

// standard library
#include <stdexcept>
#include <string>

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>

namespace DiFfRG
{
  /**
   * @brief The integers start, ..., stop - 1: forward(i) = i + start, size stop - start.
   *
   * @tparam Idx Integer type.
   */
  template <typename Idx> class IndexStack
  {
  public:
    /// Scalar type.
    using ctype = Idx;
    /// One axis.
    static constexpr size_t dim = 1;

    /// The integers [@p start, @p stop); throws if start > stop.
    IndexStack(Idx start, Idx stop) : start(start), stop(stop), m_size(stop - start)
    {
      if (start > stop)
        throw std::invalid_argument("start must be smaller than stop; start = " + std::to_string(start) +
                                    ", stop = " + std::to_string(stop));
    }

    /// [other.start, other.stop) of a LinearCoordinates1D.
    template <typename Idx2> IndexStack(const LinearCoordinates1D<Idx2> &other) : IndexStack(other.start, other.stop) {}

    /// x + start.
    Idx KOKKOS_FORCEINLINE_FUNCTION forward(const Idx x) const { return x + start; }

    /// y - start.
    Idx KOKKOS_FORCEINLINE_FUNCTION backward(const Idx y) const { return y - start; }

    /// stop - start.
    size_t size() const { return m_size; }

    /// First value, and one past the last.
    const Idx start, stop;

    /// "IndexStack(start, stop)".
    std::string to_string() const { return "IndexStack(" + std::to_string(start) + ", " + std::to_string(stop) + ")"; }

  private:
    const size_t m_size;
  };

  /**
   * @brief Bosonic Matsubara frequencies \f$\omega_n = 2\pi T n\f$ for n = start, ..., stop - 1.
   *
   * Grid index i corresponds to n = i + start: forward(i) \f$= 2\pi T (i + \mathrm{start})\f$.
   *
   * @tparam Idx Integer type of n.
   * @tparam NT Scalar type of the frequencies.
   */
  template <typename Idx = int, typename NT = double> class BosonicMatsubaraValues
  {
  public:
    /// Scalar type.
    using ctype = NT;
    /// One axis.
    static constexpr size_t dim = 1;

    /// Frequencies n = @p start, ..., @p stop - 1 at temperature @p T; throws if start > stop.
    BosonicMatsubaraValues(Idx start, Idx stop, NT T) : start(start), stop(stop), T(T), m_size(stop - start)
    {
      if (start > stop)
        throw std::invalid_argument("start must be smaller than stop; start = " + std::to_string(start) +
                                    ", stop = " + std::to_string(stop));
    }

    /// Convert from other types.
    template <typename Idx2, typename NT2>
    BosonicMatsubaraValues(const BosonicMatsubaraValues<Idx2, NT2> &other)
        : BosonicMatsubaraValues(other.start, other.stop, other.T)
    {
    }

    /// The multi-index {i} of point i.
    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t i) const
    {
      return device::array<size_t, 1>{i};
    }

    /// The frequency at grid index @p x.
    NT KOKKOS_FORCEINLINE_FUNCTION forward(const Idx &x) const { return NT(x + start) * 2. * M_PI * T; }

    /// As forward(x[0]).
    template <typename IT> device::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 1> &x) const
    {
      return {forward(x[0])};
    }

    /// The grid index of the Matsubara frequency nearest to @p y.
    Idx KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const { return Idx(std::round(y / (2. * M_PI * T))) - start; }

    /// The fractional grid index of the frequency @p y (integer on the Matsubara frequencies).
    NT KOKKOS_FORCEINLINE_FUNCTION backward_continuous(const NT &y) const { return y / (2. * M_PI * T) - NT(start); }

    /// stop - start.
    size_t size() const { return m_size; }

    /// First n, and one past the last.
    const Idx start, stop;
    /// Temperature.
    const NT T;

    /// "<Type>(start, stop, T)"; identifies the grid.
    std::string to_string() const
    {
      return "BosonicMatsubaraValues(" + std::to_string(start) + ", " + std::to_string(stop) + ", " +
             internal::round_trip(T) + ")";
    }

  private:
    const size_t m_size;
  };

  /**
   * @brief Fermionic Matsubara frequencies \f$\omega_n = 2\pi T (n + 1/2)\f$ for n = start, ..., stop - 1.
   *
   * Grid index i corresponds to n = i + start: forward(i) \f$= 2\pi T (i + \mathrm{start} + 1/2)\f$.
   *
   * @tparam Idx Integer type of n.
   * @tparam NT Scalar type of the frequencies.
   */
  template <typename Idx = int, typename NT = double> class FermionicMatsubaraValues
  {
  public:
    /// Scalar type.
    using ctype = NT;
    /// One axis.
    static constexpr size_t dim = 1;

    /// Frequencies n = @p start, ..., @p stop - 1 at temperature @p T; throws if start > stop.
    FermionicMatsubaraValues(Idx start, Idx stop, NT T) : start(start), stop(stop), T(T), m_size(stop - start)
    {
      if (start > stop)
        throw std::invalid_argument("start must be smaller than stop; start = " + std::to_string(start) +
                                    ", stop = " + std::to_string(stop));
    }

    /// Convert from other types.
    template <typename Idx2, typename NT2>
    FermionicMatsubaraValues(const FermionicMatsubaraValues<Idx2, NT2> &other)
        : FermionicMatsubaraValues(other.start, other.stop, other.T)
    {
    }

    /// The multi-index {i} of point i.
    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t i) const
    {
      return device::array<size_t, 1>{i};
    }

    /// The frequency at grid index @p x.
    NT KOKKOS_FORCEINLINE_FUNCTION forward(const Idx &x) const { return (NT(x + start) + 0.5) * 2. * M_PI * T; }

    /// As forward(x[0]).
    template <typename IT> device::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 1> &x) const
    {
      return {forward(x[0])};
    }

    /// The grid index of the Matsubara frequency nearest to @p y.
    Idx KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const
    {
      return Idx(std::round((y - M_PI * T) / (2. * M_PI * T))) - start;
    }

    /// The fractional grid index of the frequency @p y (integer on the Matsubara frequencies).
    NT KOKKOS_FORCEINLINE_FUNCTION backward_continuous(const NT &y) const
    {
      return (y - M_PI * T) / (2. * M_PI * T) - NT(start);
    }

    /// stop - start.
    size_t size() const { return m_size; }

    /// First n, and one past the last.
    const Idx start, stop;
    /// Temperature.
    const NT T;

    /// "<Type>(start, stop, T)"; identifies the grid.
    std::string to_string() const
    {
      return "FermionicMatsubaraValues(" + std::to_string(start) + ", " + std::to_string(stop) + ", " +
             internal::round_trip(T) + ")";
    }

  private:
    const size_t m_size;
  };
} // namespace DiFfRG
