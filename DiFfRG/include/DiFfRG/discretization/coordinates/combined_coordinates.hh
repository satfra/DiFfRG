#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/discretization/coordinates/stack_coordinates.hh>

namespace DiFfRG
{
  /**
   * @brief Grid of bosonic Matsubara frequencies \f$p_0 = 2\pi T n\f$ times a spatial momentum axis \f$|\vec p|\f$.
   *
   * Multi-index (m, p): frequency index m (n = m + m_start, see BosonicMatsubaraValues) and radial index p. The
   * linear index is m · grid_extent + p (frequency-major). forward(m, p) = \f$(2\pi T (m + m_{\rm start}),
   * |\vec p|(p))\f$.
   *
   * @tparam Idx Integer type of the Matsubara index.
   * @tparam NT Scalar type.
   * @tparam Radial The radial axis. LogarithmicCoordinates1D (default) is built by the seven-argument constructor;
   * any other, e.g. FocusedLogCoordinates1D, is passed as an object.
   */
  template <typename Idx = int, typename NT = double, typename Radial = LogarithmicCoordinates1D<NT>>
  class BosonicCoordinates1DFiniteT
  {
  public:
    /// Scalar type.
    using ctype = NT;
    /// Two axes: frequency and spatial momentum.
    static constexpr size_t dim = 2;
    /// The radial axis type.
    using radial_type = Radial;

    /// Frequencies n = @p m_start, ..., @p m_stop - 1 at temperature @p m_T; LogarithmicCoordinates1D(@p grid_extent,
    /// @p p_start, @p p_stop, @p p_bias) for the momentum.
    BosonicCoordinates1DFiniteT(Idx m_start, Idx m_stop, NT m_T, size_t grid_extent, NT p_start, NT p_stop, NT p_bias)
      requires std::is_same_v<Radial, LogarithmicCoordinates1D<NT>>
        : m_start(m_start), m_stop(m_stop), m_T(m_T), p_start(p_start), p_stop(p_stop), grid_extent(grid_extent),
          m_size(m_stop - m_start), matsubara_values(m_start, m_stop, m_T),
          radial_coordinates(grid_extent, p_start, p_stop, p_bias)
    {
    }

    /// Convert from other types.
    template <typename Idx2, typename NT2, typename Radial2>
    BosonicCoordinates1DFiniteT(const BosonicCoordinates1DFiniteT<Idx2, NT2, Radial2> &other)
        : BosonicCoordinates1DFiniteT(other.radial(), other.m_start, other.m_stop, other.m_T)
    {
    }

    /// Frequencies n = @p m_start, ..., @p m_stop - 1 at temperature @p m_T, and the given momentum axis.
    template <typename Radial2>
    BosonicCoordinates1DFiniteT(const Radial2 &radial_coordinates, Idx m_start, Idx m_stop, NT m_T)
        : m_start(m_start), m_stop(m_stop), m_T(m_T), p_start(radial_coordinates.start),
          p_stop(radial_coordinates.stop), grid_extent(radial_coordinates.size()), m_size(m_stop - m_start),
          matsubara_values(m_start, m_stop, m_T), radial_coordinates(radial_coordinates)
    {
    }

    /**
     * @brief The point (frequency, spatial momentum) at grid indices (@p m, @p p).
     *
     * @param m Matsubara index (0 is m_start).
     * @param p Radial grid index.
     */
    device::array<NT, 2> KOKKOS_FORCEINLINE_FUNCTION forward(const size_t m, const size_t p) const
    {
      return {matsubara_values.forward(m), radial_coordinates.forward(p)};
    }
    /// As forward(mp[0], mp[1]).
    template <typename IT> device::array<NT, 2> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 2> mp) const
    {
      return forward(mp[0], mp[1]);
    }

    /**
     * @brief Fractional grid indices of the point (@p m, @p p): frequency and spatial momentum.
     *
     * The frequency index is fractional, so interpolation blends neighbouring frequencies. If m_start ≥ 0, only
     * non-negative frequencies are stored and m is replaced by |m| (a function even in the frequency). A frequency
     * outside the stored range is mapped to the outermost stored frequency \f$m_{\rm edge}\f$ at the spatial
     * momentum \f$p'\f$ with \f$p'^2 = p^2 + m^2 - m_{\rm edge}^2\f$ (same four-momentum squared, clamped at 0).
     *
     * @param m frequency
     * @param p spatial momentum
     * @return The fractional Matsubara index and radial grid index.
     */
    std::tuple<NT, NT> KOKKOS_FUNCTION backward(NT m, const NT p) const
    {
      if (m_start >= 0) m = Kokkos::abs(m);
      const NT s = matsubara_values.backward_continuous(m);
      if (s >= NT(0) && s <= NT(m_size - 1)) return {s, radial_coordinates.backward(p)};
      const Idx m_idx = s < NT(0) ? Idx(0) : Idx(m_size) - 1;
      const NT m_edge = matsubara_values.forward(m_idx);
      // max(): a stack not containing 0 can be left towards 0, where |m| < |m_edge|
      const NT new_p = std::sqrt(Kokkos::max(NT(0), powr<2>(p) + powr<2>(m) - powr<2>(m_edge)));
      return {NT(m_idx), radial_coordinates.backward(new_p)};
    }

    /// (i / grid_extent, i % grid_extent): Matsubara and radial index of point i.
    device::array<size_t, 2> KOKKOS_INLINE_FUNCTION from_linear_index(auto i) const
    {
      device::array<size_t, 2> idx;
      idx[0] = i / grid_extent; // Matsubara index
      idx[1] = i % grid_extent; // Radial index

      return idx;
    }

    /// Number of points: number of frequencies × grid_extent.
    size_t size() const { return m_size * grid_extent; }
    /// {number of frequencies, grid_extent}.
    device::array<size_t, 2> sizes() const { return {{m_size, grid_extent}}; }

    /// The radial axis.
    const Radial &radial() const { return radial_coordinates; }

    /// Matsubara range [m_start, m_stop), temperature, and the ends of the radial axis.
    const NT m_start, m_stop, m_T, p_start, p_stop;

    /// "Combined(<frequencies>, <radial axis>)"; identifies the grid.
    std::string to_string() const
    {
      return "Combined(" + matsubara_values.to_string() + ", " + radial_coordinates.to_string() + ")";
    }

  private:
    const size_t grid_extent, m_size;

    BosonicMatsubaraValues<Idx, NT> matsubara_values;
    Radial radial_coordinates;
  };

  /**
   * @brief Grid of fermionic Matsubara frequencies \f$p_0 = 2\pi T (n + 1/2)\f$ times a spatial momentum axis; as
   * BosonicCoordinates1DFiniteT otherwise.
   */
  template <typename Idx = int, typename NT = double, typename Radial = LogarithmicCoordinates1D<NT>>
  class FermionicCoordinates1DFiniteT
  {
  public:
    /// Scalar type.
    using ctype = NT;
    /// Two axes: frequency and spatial momentum.
    static constexpr size_t dim = 2;
    /// The radial axis type.
    using radial_type = Radial;

    /// Frequencies n = @p m_start, ..., @p m_stop - 1 at temperature @p m_T; LogarithmicCoordinates1D(@p grid_extent,
    /// @p p_start, @p p_stop, @p p_bias) for the momentum.
    FermionicCoordinates1DFiniteT(Idx m_start, Idx m_stop, NT m_T, size_t grid_extent, NT p_start, NT p_stop, NT p_bias)
      requires std::is_same_v<Radial, LogarithmicCoordinates1D<NT>>
        : m_start(m_start), m_stop(m_stop), m_T(m_T), p_start(p_start), p_stop(p_stop), grid_extent(grid_extent),
          m_size(m_stop - m_start), matsubara_values(m_start, m_stop, m_T),
          radial_coordinates(grid_extent, p_start, p_stop, p_bias)
    {
    }

    /// Convert from other types.
    template <typename Idx2, typename NT2, typename Radial2>
    FermionicCoordinates1DFiniteT(const FermionicCoordinates1DFiniteT<Idx2, NT2, Radial2> &other)
        : FermionicCoordinates1DFiniteT(other.radial(), other.m_start, other.m_stop, other.m_T)
    {
    }

    /// Frequencies n = @p m_start, ..., @p m_stop - 1 at temperature @p m_T, and the given momentum axis.
    template <typename Radial2>
    FermionicCoordinates1DFiniteT(const Radial2 &radial_coordinates, Idx m_start, Idx m_stop, NT m_T)
        : m_start(m_start), m_stop(m_stop), m_T(m_T), p_start(radial_coordinates.start),
          p_stop(radial_coordinates.stop), grid_extent(radial_coordinates.size()), m_size(m_stop - m_start),
          matsubara_values(m_start, m_stop, m_T), radial_coordinates(radial_coordinates)
    {
    }

    /**
     * @brief The point (frequency, spatial momentum) at grid indices (@p m, @p p).
     *
     * @param m Matsubara index (0 is m_start).
     * @param p Radial grid index.
     */
    device::array<NT, 2> KOKKOS_FORCEINLINE_FUNCTION forward(const size_t m, const size_t p) const
    {
      return {matsubara_values.forward(m), radial_coordinates.forward(p)};
    }
    /// As forward(mp[0], mp[1]).
    template <typename IT> device::array<NT, 2> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 2> mp) const
    {
      return forward(mp[0], mp[1]);
    }

    /**
     * @brief Fractional grid indices of the point (@p m, @p p), as BosonicCoordinates1DFiniteT::backward but without
     * replacing m by |m| (at nonzero chemical potential the function is not even in the frequency).
     *
     * @param m frequency
     * @param p spatial momentum
     * @return The fractional Matsubara index and radial grid index.
     */
    std::tuple<NT, NT> KOKKOS_FUNCTION backward(const NT m, const NT p) const
    {
      const NT s = matsubara_values.backward_continuous(m);
      if (s >= NT(0) && s <= NT(m_size - 1)) return {s, radial_coordinates.backward(p)};
      const Idx m_idx = s < NT(0) ? Idx(0) : Idx(m_size) - 1;
      const NT m_edge = matsubara_values.forward(m_idx);
      // max(): a stack not containing 0 can be left towards 0, where |m| < |m_edge|
      const NT new_p = std::sqrt(Kokkos::max(NT(0), powr<2>(p) + powr<2>(m) - powr<2>(m_edge)));
      return {NT(m_idx), radial_coordinates.backward(new_p)};
    }

    /// (i / grid_extent, i % grid_extent): Matsubara and radial index of point i.
    device::array<size_t, 2> KOKKOS_INLINE_FUNCTION from_linear_index(auto i) const
    {
      device::array<size_t, 2> idx;
      idx[0] = i / grid_extent; // Matsubara index
      idx[1] = i % grid_extent; // Radial index

      return idx;
    }

    /// Number of points: number of frequencies × grid_extent.
    size_t size() const { return m_size * grid_extent; }
    /// {number of frequencies, grid_extent}.
    device::array<size_t, 2> sizes() const { return {{m_size, grid_extent}}; }

    /// The radial axis.
    const Radial &radial() const { return radial_coordinates; }

    /// Matsubara range [m_start, m_stop), temperature, and the ends of the radial axis.
    const NT m_start, m_stop, m_T, p_start, p_stop;

    /// "Combined(<frequencies>, <radial axis>)"; identifies the grid.
    std::string to_string() const
    {
      return "Combined(" + matsubara_values.to_string() + ", " + radial_coordinates.to_string() + ")";
    }

  private:
    const size_t grid_extent, m_size;

    FermionicMatsubaraValues<Idx, NT> matsubara_values;
    Radial radial_coordinates;
  };

  // Finite-T coordinates whose radial axis is focused on an interior scale
  using FocusedBosonicCoordinates1DFiniteT = BosonicCoordinates1DFiniteT<int, double, FocusedLogCoordinates1D<double>>;
  using FocusedFermionicCoordinates1DFiniteT =
      FermionicCoordinates1DFiniteT<int, double, FocusedLogCoordinates1D<double>>;

  /// See rebind_ctype in coordinates.hh.
  template <typename Idx, typename NT0, typename Radial, typename NT>
    requires requires { typename rebind_ctype<Radial, NT>::type; }
  struct rebind_ctype<BosonicCoordinates1DFiniteT<Idx, NT0, Radial>, NT> {
    using type = BosonicCoordinates1DFiniteT<Idx, NT, rebind_ctype_t<Radial, NT>>;
  };
  template <typename Idx, typename NT0, typename Radial, typename NT>
    requires requires { typename rebind_ctype<Radial, NT>::type; }
  struct rebind_ctype<FermionicCoordinates1DFiniteT<Idx, NT0, Radial>, NT> {
    using type = FermionicCoordinates1DFiniteT<Idx, NT, rebind_ctype_t<Radial, NT>>;
  };
} // namespace DiFfRG
