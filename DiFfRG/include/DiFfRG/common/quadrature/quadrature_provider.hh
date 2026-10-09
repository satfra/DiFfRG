#pragma once

// DiFfRG
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>
#include <DiFfRG/common/run_reporter.hh>

// standard library
#include <mutex>
#include <optional>

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

      /**
       * @brief The exact Matsubara sum for a summand of finite extent in the frequency.
       *
       * Keyed on (T, node count) rather than on the cutoff, because that is all the rule depends
       * on: the cutoff enters only through floor(cutoff / 2 pi T). Neighbouring RG steps therefore
       * share an entry instead of each building their own, and the map stays bounded by the number
       * of distinct mode counts the flow visits.
       */
      template <typename NT = double> MatsubaraQuadrature<NT> &get_matsubara_exact_sum(const NT T, const NT freq_cutoff)
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        const int n = MatsubaraQuadrature<NT>::modes_below(T, freq_cutoff);

        if constexpr (std::is_same_v<NT, double>) {
          return find_exact_d(n, find_exact_T_d(T));
        } else if constexpr (std::is_same_v<NT, float>) {
          return find_exact_f(n, find_exact_T_f(T));
        }
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                      "Unknown type requested of MatsubaraStorage::get_matsubara_exact_sum");
      }

      /**
       * @brief Gauss-Legendre over [-cutoff, cutoff] for a summand of finite extent; see
       * MatsubaraQuadrature::reinit_finite_interval.
       *
       * Keyed on (order, cutoff). The cutoff is continuous -- it tracks k -- so this rebuilds once
       * per RG step, exactly as the T=0 rule it replaces already did. The Gauss-Legendre rule
       * itself is NOT rebuilt: it comes from QuadratureStorage and only the O(N) scaling by the
       * cutoff happens here.
       */
      template <typename NT = double>
      MatsubaraQuadrature<NT> &get_finite_interval(const NT cutoff, const size_t order,
                                                   const Kokkos::View<const NT *, CPU_memory> gl_nodes,
                                                   const Kokkos::View<const NT *, CPU_memory> gl_weights)
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        if constexpr (std::is_same_v<NT, double>) {
          return find_interval_d(cutoff, find_interval_order_d(order), gl_nodes, gl_weights);
        } else if constexpr (std::is_same_v<NT, float>) {
          return find_interval_f(cutoff, find_interval_order_f(order), gl_nodes, gl_weights);
        }
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                      "Unknown type requested of MatsubaraStorage::get_finite_interval");
      }

      void set_verbosity(int v);
      void set_report_port(ReportPort port) { log = std::move(port); }

      void set_vacuum_quad_size(const int size);
      void set_min_matsubara_size(const int value);
      void set_max_matsubara_size(const int value);
      void set_matsubara_precision_factor(const double value);

      /// The node ceiling every rule built here is clamped to.
      int get_max_matsubara_size() const { return max_matsubara_size; }

      /**
       * @brief Nodes the Monien rule would want at (T, typical_E) under THIS provider's dials.
       *
       * Exposed because choosing between a sum and an integral is the caller's decision (see
       * QuadratureIntegrator_fT::refresh_matsubara) but the numbers that price it -- the reach
       * multiplier and matsubara_precision_factor -- live here.
       */
      template <typename NT = double> int predicted_size(const NT T, const NT typical_E) const
      {
        return MatsubaraQuadrature<NT>::predict_size(T, typical_E, matsubara_precision_factor);
      }

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

      template <typename NT = double> using ExactStorageType = std::map<double, std::map<int, MatsubaraQuadrature<NT>>>;
      template <typename NT = double> using ExactTemperatureIterator = typename ExactStorageType<NT>::iterator;

      ExactTemperatureIterator<double> find_exact_T_d(const double T);
      ExactTemperatureIterator<float> find_exact_T_f(const float T);

      MatsubaraQuadrature<double> &find_exact_d(const int n, ExactTemperatureIterator<double> T_it);
      MatsubaraQuadrature<float> &find_exact_f(const int n, ExactTemperatureIterator<float> T_it);

      StorageType<double> quadratures_d;
      StorageType<float> quadratures_f;

      ExactStorageType<double> exact_sums_d;
      ExactStorageType<float> exact_sums_f;

      template <typename NT = double>
      using IntervalStorageType = std::map<size_t, std::map<double, MatsubaraQuadrature<NT>>>;
      template <typename NT = double> using IntervalOrderIterator = typename IntervalStorageType<NT>::iterator;

      IntervalOrderIterator<double> find_interval_order_d(const size_t order);
      IntervalOrderIterator<float> find_interval_order_f(const size_t order);

      MatsubaraQuadrature<double> &find_interval_d(const double cutoff, IntervalOrderIterator<double> o_it,
                                                   const Kokkos::View<const double *, CPU_memory> n,
                                                   const Kokkos::View<const double *, CPU_memory> w);
      MatsubaraQuadrature<float> &find_interval_f(const float cutoff, IntervalOrderIterator<float> o_it,
                                                  const Kokkos::View<const float *, CPU_memory> n,
                                                  const Kokkos::View<const float *, CPU_memory> w);

      IntervalStorageType<double> intervals_d;
      IntervalStorageType<float> intervals_f;

      int verbosity = 0;
      // These must agree with the ConfigTree defaults read in QuadratureProvider's
      // ConfigTree constructor, otherwise a default-constructed provider (every test, and any
      // integrator built without a config) silently runs a different rule than production.
      int vacuum_quad_size = 64;
      double matsubara_precision_factor = 1;
      int min_matsubara_size = 8;
      int max_matsubara_size = 128;

      std::mutex m_mutex;
      ReportPort log;
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
      void set_report_port(ReportPort port) { log = std::move(port); }

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
      ReportPort log;
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
    /**
     * @brief Construct a provider that reports the quadratures it builds.
     *
     * The standalone overload reports into the run log, <output folder>/<output name>.log, under the
     * "quadrature" tag when /output/folder is present. It owns that reporter rather than borrow one
     * from an OutputSession, because integrators request their quadratures inside their constructors
     * -- typically before any session exists; the file sink is shared process-wide per path, so the
     * session opening the same file later appends to the quadrature inventory instead of truncating
     * it. The inventory never echoes to the console, where it would drown the timestepper progress
     * report. Passing a port joins an existing run reporter.
     */
    explicit QuadratureProvider(const ConfigTree &config);
    QuadratureProvider(const ConfigTree &config, ReportPort log);

    /**
     * @brief Get the quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double, typename MemorySpace = CPU_memory>
    auto nodes(const size_t order, const QuadratureType type = QuadratureType::legendre)
    {
      return quadrature_storage.get_quadrature<NT>(order, type).template nodes<MemorySpace>();
    }

    /**
     * @brief Get the quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double, typename MemorySpace = CPU_memory>
    auto weights(const size_t order, const QuadratureType type = QuadratureType::legendre)
    {
      return quadrature_storage.get_quadrature<NT>(order, type).template weights<MemorySpace>();
    }

    /**
     * @brief Get the quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double, typename MemorySpace = CPU_memory>
    auto matsubara_nodes(const NT T, const NT typical_E)
    {
      return matsubara_storage.get_matsubara_quadrature<NT>(T, typical_E).template nodes<MemorySpace>();
    }

    /**
     * @brief Get the quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double, typename MemorySpace = CPU_memory>
    auto matsubara_weights(const NT T, const NT typical_E)
    {
      return matsubara_storage.get_matsubara_quadrature<NT>(T, typical_E).template weights<MemorySpace>();
    }

    /**
     * @brief The Matsubara rule itself, for callers that need more than nodes and weights -- its
     * size, or the zero-mode-inclusive node list of sum_nodes(). The reference is into a
     * node-based map and stays valid for the lifetime of the provider.
     */
    template <typename NT = double> const MatsubaraQuadrature<NT> &matsubara_rule(const NT T, const NT typical_E)
    {
      return matsubara_storage.get_matsubara_quadrature<NT>(T, typical_E);
    }

    /// The node ceiling /integration/max_matsubara_size, i.e. the budget the rule choice is made against.
    int max_matsubara_size() const { return matsubara_storage.get_max_matsubara_size(); }

    /**
     * @brief Nodes the Monien rule would want at (T, typical_E). See MatsubaraStorage::predicted_size.
     */
    template <typename NT = double> int matsubara_predicted_size(const NT T, const NT typical_E) const
    {
      return matsubara_storage.template predicted_size<NT>(T, typical_E);
    }

    /**
     * @brief The exact Matsubara sum for a summand that vanishes above `freq_cutoff`.
     * See MatsubaraQuadrature::reinit_exact_sum.
     */
    template <typename NT = double> const MatsubaraQuadrature<NT> &matsubara_exact_sum(const NT T, const NT freq_cutoff)
    {
      return matsubara_storage.get_matsubara_exact_sum<NT>(T, freq_cutoff);
    }

    /**
     * @brief Gauss-Legendre over the finite frequency interval a compactly supported summand lives
     * on, at the given order. See MatsubaraQuadrature::reinit_finite_interval.
     */
    template <typename NT = double>
    const MatsubaraQuadrature<NT> &matsubara_finite_interval(const NT cutoff, const size_t order)
    {
      auto &gl = quadrature_storage.get_quadrature<NT>(order, QuadratureType::legendre);
      return matsubara_storage.template get_finite_interval<NT>(cutoff, order, gl.template nodes<CPU_memory>(),
                                                                gl.template weights<CPU_memory>());
    }

    /**
     * @brief Hand out one of a small pool of execution space instances, round robin.
     *
     * Every integrator used to default-construct its ExecutionSpace, i.e. they all shared the
     * default stream and no two flows could ever execute concurrently -- even the ones that are
     * mutually independent (the vertex flows of a typical RHS read the same dressings and write
     * disjoint outputs). Handing neighbouring integrators different instances lets those overlap,
     * which matters most for the many small launches that individually leave the device
     * underutilised.
     *
     * Safe to mix with the interpolators' uploads, which fence on the host before returning, so
     * any kernel issued afterwards on any stream already sees the new data.
     *
     * The pool is created lazily and shared by every integrator holding this provider. Host
     * execution spaces have no streams to speak of, so they get a default-constructed instance.
     */
    template <typename ExecutionSpace> ExecutionSpace next_execution_space()
    {
      if constexpr (std::is_same_v<typename ExecutionSpace::memory_space, CPU_memory>) {
        return ExecutionSpace();
      } else {
        static constexpr size_t pool_size = 8;
        static std::vector<ExecutionSpace> pool = [] {
          const std::vector<int> weights(pool_size, 1);
          return Kokkos::Experimental::partition_space(ExecutionSpace(), weights);
        }();
        static size_t next = 0;
        return pool[next++ % pool_size];
      }
    }

  private:
    void initialize(const ConfigTree &config, ReportPort log);

    std::optional<RunReporter> own_logger;
    internal::MatsubaraStorage matsubara_storage;
    internal::QuadratureStorage quadrature_storage;

    int verbosity;
  };
} // namespace DiFfRG
