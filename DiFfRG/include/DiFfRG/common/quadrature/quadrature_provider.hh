#pragma once

// DiFfRG
#include "DiFfRG/common/json.hh"
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>

// standard library
#include <mutex>
#include <vector>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief A class that stores Matsubara quadrature points and weights for a given T, E.
     * Its main purpose is to avoid recomputing the quadrature points and weights for each Matsubara integrator and
     * provide a search algorithm to find previously computed Matsubara quadratures.
     */
    class MatsubaraStorage
    {
    public:
      /**
       * @brief Return the MatsubaraQuadrature object for a given T, E
       *
       */
      template <typename NT = double> MatsubaraQuadrature<NT> &get_matsubara_quadrature(const NT T, const NT E)
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        if constexpr (std::is_same_v<NT, double>) {
          auto T_it = find_T_d(T);
          auto E_it = find_E_d(E, T_it);
          return E_it->second;
        } else if constexpr (std::is_same_v<NT, float>) {
          auto T_it = find_T_f(T);
          auto E_it = find_E_f(E, T_it);
          return E_it->second;
        }
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                      "Unknown type requested of MatsubaraStorage::get_matsubara_quadrature");
      }

      void set_verbosity(int v);

      void set_vacuum_quad_size(const int size);
      void set_min_matsubara_size(const int value);
      void set_add_matsubara_size(const int value);

    private:
      MatsubaraQuadrature<double> &get_matsubara_quadrature_d(const double T, const double E);
      MatsubaraQuadrature<float> &get_matsubara_quadrature_f(const float T, const float E);

      template <typename NT = double> using StorageType = std::map<double, std::map<double, MatsubaraQuadrature<NT>>>;
      template <typename NT = double> using SubStorageType = std::map<double, MatsubaraQuadrature<NT>>;

      template <typename NT = double> using TemperatureIterator = typename StorageType<NT>::iterator;
      template <typename NT = double> using EnergyIterator = typename StorageType<NT>::mapped_type::iterator;

      TemperatureIterator<double> find_T_d(const double T);
      TemperatureIterator<float> find_T_f(const float T);

      EnergyIterator<double> find_E_d(const double E, TemperatureIterator<double> T_it);
      EnergyIterator<float> find_E_f(const float E, TemperatureIterator<float> T_it);

      StorageType<double> quadratures_d;
      StorageType<float> quadratures_f;

      int verbosity = 0;
      int vacuum_quad_size = 48;
      int add_matsubara_size = 0;
      int min_matsubara_size = 32;

      std::mutex m_mutex;
    };

    /**
     * @brief A class that stores Quadrature points and weights for a given type and order
     * Its main purpose is to avoid recomputing the quadrature points and weights for each integrator and
     * provide a search algorithm to find previously computed quadratures.
     */
    class QuadratureStorage
    {
    public:
      template <typename NT = double> Quadrature<NT> &get_quadrature(const size_t order, const QuadratureType type)
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        if constexpr (std::is_same_v<NT, double>) {
          auto type_it = find_type_d(type);
          auto order_it = find_order_d(order, type_it);
          return order_it->second;
        } else if constexpr (std::is_same_v<NT, float>) {
          auto type_it = find_type_f(type);
          auto order_it = find_order_f(order, type_it);
          return order_it->second;
        }
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                      "Unknown type requested of QuadratureStorage::get_quadrature");
      }

      void set_verbosity(int v);

    private:
      Quadrature<double> &get_quadrature_d(const size_t order, const QuadratureType type);
      Quadrature<float> &get_quadrature_f(const size_t order, const QuadratureType type);

      template <typename NT = double> using StorageType = std::map<QuadratureType, std::map<size_t, Quadrature<NT>>>;
      template <typename NT = double> using SubStorageType = std::map<size_t, Quadrature<NT>>;

      template <typename NT = double> using TypeIterator = typename StorageType<NT>::iterator;
      template <typename NT = double> using OrderIterator = typename StorageType<NT>::mapped_type::iterator;

      TypeIterator<double> find_type_d(const QuadratureType type);
      TypeIterator<float> find_type_f(const QuadratureType type);

      OrderIterator<double> find_order_d(const size_t order, TypeIterator<double> type_it);
      OrderIterator<float> find_order_f(const size_t order, TypeIterator<float> type_it);

      StorageType<double> quadratures_d;
      StorageType<float> quadratures_f;

      int verbosity = 0;

      std::mutex m_mutex;
    };
  } // namespace internal

  /**
   * @brief A class that provides quadrature points and weights, in host and device memory.
   * The quadrature points and weights are computed either the GSL quadratures or the MatsubaraQuadrature class.
   * This avoids recomputing the quadrature points and weights for each integrator.
   */
  class QuadratureProvider
  {
  public:
    QuadratureProvider();
    QuadratureProvider(const JSONValue &json);

    /**
     * @brief Get the quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double>
    const std::vector<NT> &get_points(const size_t order, const QuadratureType type = QuadratureType::legendre)
    {
      return quadrature_storage.get_quadrature<NT>(order, type).nodes();
    }

    /**
     * @brief Get the quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double>
    const std::vector<NT> &get_weights(const size_t order, const QuadratureType type = QuadratureType::legendre)
    {
      return quadrature_storage.get_quadrature<NT>(order, type).weights();
    }

    /**
     * @brief Get the quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double> const std::vector<NT> &get_matsubara_points(const NT T, const NT typical_E)
    {
      return matsubara_storage.get_matsubara_quadrature<NT>(T, typical_E).nodes();
    }

    /**
     * @brief Get the quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double> const std::vector<NT> &get_matsubara_weights(const NT T, const NT typical_E)
    {
      return matsubara_storage.get_matsubara_quadrature<NT>(T, typical_E).weights();
    }

#ifdef USE_CUDA
    /**
     * @brief Get the device-side quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const double*
     */
    template <typename NT = double>
    const NT *get_device_points(const size_t order, const int device = 0,
                                const QuadratureType type = QuadratureType::legendre)
    {
      return quadrature_storage.get_quadrature<NT>(order, type).device_nodes();
    }

    /**
     * @brief Get the device-side quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const double*
     */
    template <typename NT = double>
    const NT *get_device_weights(const size_t order, const int device = 0,
                                 const QuadratureType type = QuadratureType::legendre)

    {
      return quadrature_storage.get_quadrature<NT>(order, type).device_weights();
    }

    /**
     * @brief Get the device-side quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const double*
     */
    template <typename NT = double>
    const NT *get_device_matsubara_points(const NT T, const NT typical_E, const int device = 0)
    {
      return matsubara_storage.get_matsubara_quadrature<NT>(T, typical_E).device_nodes();
    }

    /**
     * @brief Get the device-side quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const double*
     */
    template <typename NT = double>
    const NT *get_device_matsubara_weights(const NT T, const NT typical_E, const int device = 0)
    {
      return matsubara_storage.get_matsubara_quadrature<NT>(T, typical_E).device_weights();
    }
#endif

  private:
    internal::MatsubaraStorage matsubara_storage;
    internal::QuadratureStorage quadrature_storage;

    int verbosity;
  };
} // namespace DiFfRG