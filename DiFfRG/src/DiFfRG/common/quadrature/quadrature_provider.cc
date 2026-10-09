// DiFfRG
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  namespace internal
  {
    MatsubaraStorage::TemperatureIterator<double> MatsubaraStorage::find_T_d(const double T)
    {
      // search for T
      auto it = quadratures_d.lower_bound(T);

      if (it != quadratures_d.end()) {
        bool it_found_T = (is_close(it->first, T, 1e-6 * T));

        if (it_found_T) return it;

        if (it != quadratures_d.begin()) {
          bool itp_found_T = (std::prev(it) != quadratures_d.end()) && (is_close(std::prev(it)->first, T, 1e-6 * T));
          if (itp_found_T) return std::prev(it);
        }
      }

      // create new entry in quadratures_d
      return quadratures_d.insert(std::make_pair(T, SubStorageType<double>())).first;
    }

    MatsubaraStorage::EnergyIterator<double> MatsubaraStorage::find_E_d(const double E,
                                                                        TemperatureIterator<double> T_it)
    {
      auto &map = T_it->second;

      // search for E
      auto it = map.lower_bound(E);

      if (it != map.end()) {
        bool it_found_T = (is_close(it->first, E, 1e-6 * E));

        if (it_found_T) return it;

        if (it != map.begin()) {
          bool itp_found_T = (std::prev(it) != map.end()) && (is_close(std::prev(it)->first, E, 1e-6 * E));
          if (itp_found_T) return std::prev(it);
        }
      }

      // create new entry in quadratures_d
      auto new_it = map.insert(std::make_pair(E, MatsubaraQuadrature<double>())).first;
      // Initialize the quadrature
      new_it->second.reinit(T_it->first, E, 2, min_matsubara_size, max_matsubara_size, vacuum_quad_size,
                            matsubara_precision_factor);

      if (verbosity >= 0)
        log.info("Created MatsubaraQuadrature<double> with T = {:.4} and E = {:.4} and size = {}", T_it->first, E,
                 new_it->second.size());

      return new_it;
    }

    MatsubaraStorage::TemperatureIterator<float> MatsubaraStorage::find_T_f(const float T)
    {
      // search for T
      auto it = quadratures_f.lower_bound(T);

      if (it != quadratures_f.end()) {
        bool it_found_T = (is_close(it->first, T, 1e-6 * T));

        if (it_found_T) return it;

        if (it != quadratures_f.begin()) {
          bool itp_found_T = (std::prev(it) != quadratures_f.end()) && (is_close(std::prev(it)->first, T, 1e-6 * T));
          if (itp_found_T) return std::prev(it);
        }
      }

      // create new entry in quadratures_f
      return quadratures_f.insert(std::make_pair(T, SubStorageType<float>())).first;
    }

    MatsubaraStorage::EnergyIterator<float> MatsubaraStorage::find_E_f(const float E, TemperatureIterator<float> T_it)
    {
      auto &map = T_it->second;

      // search for E
      auto it = map.lower_bound(E);

      if (it != map.end()) {
        bool it_found_T = (is_close(it->first, E, 1e-6 * E));

        if (it_found_T) return it;

        if (it != map.begin()) {
          bool itp_found_T = (std::prev(it) != map.end()) && (is_close(std::prev(it)->first, E, 1e-6 * E));
          if (itp_found_T) return std::prev(it);
        }
      }

      // create new entry in quadratures_f
      auto new_it = map.insert(std::make_pair(E, MatsubaraQuadrature<float>())).first;
      // Initialize the quadrature
      new_it->second.reinit(T_it->first, E, 2, min_matsubara_size, max_matsubara_size, vacuum_quad_size,
                            matsubara_precision_factor);

      if (verbosity >= 0)
        log.info("Created MatsubaraQuadrature<float> with T = {:.4} and E = {:.4} and size = {}", T_it->first, E,
                 new_it->second.size());

      return new_it;
    }

    MatsubaraStorage::ExactTemperatureIterator<double> MatsubaraStorage::find_exact_T_d(const double T)
    {
      auto it = exact_sums_d.lower_bound(T);

      if (it != exact_sums_d.end()) {
        if (is_close(it->first, T, 1e-6 * T)) return it;
        if (it != exact_sums_d.begin() && is_close(std::prev(it)->first, T, 1e-6 * T)) return std::prev(it);
      }

      return exact_sums_d.insert(std::make_pair(T, std::map<int, MatsubaraQuadrature<double>>())).first;
    }

    MatsubaraQuadrature<double> &MatsubaraStorage::find_exact_d(const int n, ExactTemperatureIterator<double> T_it)
    {
      auto &map = T_it->second;

      // Integer key: an exact lookup, unlike the (T, E) maps above, which is the whole point of
      // keying on the node count instead of the continuous cutoff.
      auto it = map.find(n);
      if (it != map.end()) return it->second;

      auto new_it = map.insert(std::make_pair(n, MatsubaraQuadrature<double>())).first;
      // 2 pi (n + 1/2) T is the midpoint between mode n and mode n+1, so it reproduces exactly n
      // positive modes -- reinit_exact_sum re-derives the count from the cutoff it is given.
      new_it->second.reinit_exact_sum(T_it->first, 2. * M_PI * (n + 0.5) * T_it->first);

      if (verbosity >= 0)
        log.info("Created exact MatsubaraQuadrature<double> sum with T = {:.4} and {} positive modes", T_it->first, n);

      return new_it->second;
    }

    MatsubaraStorage::ExactTemperatureIterator<float> MatsubaraStorage::find_exact_T_f(const float T)
    {
      auto it = exact_sums_f.lower_bound(T);

      if (it != exact_sums_f.end()) {
        if (is_close(it->first, (double)T, 1e-6 * T)) return it;
        if (it != exact_sums_f.begin() && is_close(std::prev(it)->first, (double)T, 1e-6 * T)) return std::prev(it);
      }

      return exact_sums_f.insert(std::make_pair((double)T, std::map<int, MatsubaraQuadrature<float>>())).first;
    }

    MatsubaraQuadrature<float> &MatsubaraStorage::find_exact_f(const int n, ExactTemperatureIterator<float> T_it)
    {
      auto &map = T_it->second;

      auto it = map.find(n);
      if (it != map.end()) return it->second;

      auto new_it = map.insert(std::make_pair(n, MatsubaraQuadrature<float>())).first;
      new_it->second.reinit_exact_sum((float)T_it->first, (float)(2. * M_PI * (n + 0.5) * T_it->first));

      if (verbosity >= 0)
        log.info("Created exact MatsubaraQuadrature<float> sum with T = {:.4} and {} positive modes", T_it->first, n);

      return new_it->second;
    }

    MatsubaraStorage::IntervalOrderIterator<double> MatsubaraStorage::find_interval_order_d(const size_t order)
    {
      auto it = intervals_d.find(order);
      if (it != intervals_d.end()) return it;
      return intervals_d.insert(std::make_pair(order, std::map<double, MatsubaraQuadrature<double>>())).first;
    }

    MatsubaraQuadrature<double> &MatsubaraStorage::find_interval_d(const double cutoff,
                                                                   IntervalOrderIterator<double> o_it,
                                                                   const Kokkos::View<const double *, CPU_memory> n,
                                                                   const Kokkos::View<const double *, CPU_memory> w)
    {
      auto &map = o_it->second;

      auto it = map.lower_bound(cutoff);
      if (it != map.end()) {
        if (is_close(it->first, cutoff, 1e-6 * cutoff)) return it->second;
        if (it != map.begin() && is_close(std::prev(it)->first, cutoff, 1e-6 * cutoff)) return std::prev(it)->second;
      }

      auto new_it = map.insert(std::make_pair(cutoff, MatsubaraQuadrature<double>())).first;
      new_it->second.reinit_finite_interval(cutoff, n, w);

      if (verbosity >= 0)
        log.info("Created finite-interval MatsubaraQuadrature<double> with cutoff = {:.4} and order = {}", cutoff,
                 o_it->first);

      return new_it->second;
    }

    MatsubaraStorage::IntervalOrderIterator<float> MatsubaraStorage::find_interval_order_f(const size_t order)
    {
      auto it = intervals_f.find(order);
      if (it != intervals_f.end()) return it;
      return intervals_f.insert(std::make_pair(order, std::map<double, MatsubaraQuadrature<float>>())).first;
    }

    MatsubaraQuadrature<float> &MatsubaraStorage::find_interval_f(const float cutoff, IntervalOrderIterator<float> o_it,
                                                                  const Kokkos::View<const float *, CPU_memory> n,
                                                                  const Kokkos::View<const float *, CPU_memory> w)
    {
      auto &map = o_it->second;

      auto it = map.lower_bound((double)cutoff);
      if (it != map.end()) {
        if (is_close(it->first, (double)cutoff, 1e-6 * cutoff)) return it->second;
        if (it != map.begin() && is_close(std::prev(it)->first, (double)cutoff, 1e-6 * cutoff))
          return std::prev(it)->second;
      }

      auto new_it = map.insert(std::make_pair((double)cutoff, MatsubaraQuadrature<float>())).first;
      new_it->second.reinit_finite_interval(cutoff, n, w);

      if (verbosity >= 0)
        log.info("Created finite-interval MatsubaraQuadrature<float> with cutoff = {:.4} and order = {}", cutoff,
                 o_it->first);

      return new_it->second;
    }

    void MatsubaraStorage::set_verbosity(int v) { verbosity = v; }
    void MatsubaraStorage::set_vacuum_quad_size(const int size)
    {
      if (size <= 0) throw std::invalid_argument("MatsubaraStorage: Vacuum quadrature size must be positive.");
      vacuum_quad_size = size;
    }
    void MatsubaraStorage::set_matsubara_precision_factor(const double value)
    {
      if (value < 0) throw std::invalid_argument("MatsubaraStorage: matsubara_precision_factor must be positive.");
      matsubara_precision_factor = value;
    }
    void MatsubaraStorage::set_min_matsubara_size(const int value)
    {
      if (value <= 0) throw std::invalid_argument("MatsubaraStorage: min_matsubara_size must be positive.");
      min_matsubara_size = value;
    }
    void MatsubaraStorage::set_max_matsubara_size(const int value)
    {
      if (value <= 0) throw std::invalid_argument("MatsubaraStorage: max_matsubara_size must be positive.");
      max_matsubara_size = value;
    }

    QuadratureStorage::TypeIterator<double> QuadratureStorage::find_type_d(const QuadratureType type)
    {
      // search for type
      auto it = quadratures_d.find(type);

      if (it != quadratures_d.end()) return it;

      // create new entry in quadratures_d
      return quadratures_d.insert(std::make_pair(type, SubStorageType<double>())).first;
    }

    QuadratureStorage::OrderIterator<double> QuadratureStorage::find_order_d(const size_t order,
                                                                             TypeIterator<double> type_it)
    {
      auto &map = type_it->second;

      // search for T in temperatures_d
      auto it = map.find(order);

      if (it != map.end()) return it;

      // create new entry in quadratures_d
      auto new_it = map.insert(std::make_pair(order, Quadrature<double>())).first;

      // Initialize the quadrature
      new_it->second.reinit(order, type_it->first);

      if (verbosity >= 0)
        log.info("Created Quadrature<double> with order = {0} and type = {1}", order, (uint)type_it->first.kind);

      return new_it;
    }

    QuadratureStorage::TypeIterator<float> QuadratureStorage::find_type_f(const QuadratureType type)
    {
      // search for type
      auto it = quadratures_f.find(type);

      if (it != quadratures_f.end()) return it;

      // create new entry in quadratures_f
      return quadratures_f.insert(std::make_pair(type, SubStorageType<float>())).first;
    }

    QuadratureStorage::OrderIterator<float> QuadratureStorage::find_order_f(const size_t order,
                                                                            TypeIterator<float> type_it)
    {
      auto &map = type_it->second;

      // search for order
      auto it = map.find(order);

      if (it != map.end()) return it;

      // create new entry in quadratures_f
      auto new_it = map.insert(std::make_pair(order, Quadrature<float>())).first;

      // Initialize the quadrature
      new_it->second.reinit(order, type_it->first);

      if (verbosity >= 0)
        log.info("Created Quadrature<float> with order = {0} and type = {1}", order, (uint)type_it->first.kind);

      return new_it;
    }

    void QuadratureStorage::set_verbosity(int v) { verbosity = v; }
  } // namespace internal

  QuadratureProvider::QuadratureProvider()
  {
    if (!DiFfRG::Init::is_initialized()) throw std::runtime_error("QuadratureProvider: DiFfRG is not initialized.");

    verbosity = -1;
    matsubara_storage.set_verbosity(verbosity);
    quadrature_storage.set_verbosity(verbosity);
  }

  QuadratureProvider::QuadratureProvider(const ConfigTree &config)
  {
    if (!DiFfRG::Init::is_initialized()) throw std::runtime_error("QuadratureProvider: DiFfRG is not initialized.");

    if (config.contains("/output/folder")) {
      // Written into the shared run log under the "quadrature" tag; the reporter has to be owned
      // here rather than borrowed from an OutputSession, because integrators request their
      // quadratures inside their constructors -- typically before any session exists. No console
      // echo: the quadrature inventory would drown the timestepper progress report.
      own_logger.emplace(OutputPath(config), Config::OutputSettings(config), MPI::rank(MPI_COMM_WORLD) == 0,
                         RunReporterOptions{.reporter_name = "quadrature", .console = false});
      initialize(config, own_logger->port());
    } else {
      // No folder means no file destination, and the quadrature inventory never goes to the
      // console; a default RunReporter's port drops every message.
      initialize(config, RunReporter().port());
    }
  }

  QuadratureProvider::QuadratureProvider(const ConfigTree &config, ReportPort log)
  {
    if (!DiFfRG::Init::is_initialized()) throw std::runtime_error("QuadratureProvider: DiFfRG is not initialized.");
    initialize(config, std::move(log));
  }

  void QuadratureProvider::initialize(const ConfigTree &config, ReportPort log)
  {
    const Config::OutputSettings output_settings(config);
    verbosity = output_settings.verbosity;

    if (verbosity >= 0) log.info("QuadratureProvider: Initialized quadrature provider.");

    matsubara_storage.set_verbosity(verbosity);
    matsubara_storage.set_report_port(log);
    quadrature_storage.set_verbosity(verbosity);
    quadrature_storage.set_report_port(log);

    const int vacuum_quad_size = config.get_uint("/integration/vacuum_quad_size", 64);
    matsubara_storage.set_vacuum_quad_size(vacuum_quad_size);

    const double matsubara_precision_factor = config.get_double("/integration/matsubara_precision_factor", 1.);
    matsubara_storage.set_matsubara_precision_factor(matsubara_precision_factor);

    const int min_matsubara_size = config.get_uint("/integration/min_matsubara_size", 8);
    matsubara_storage.set_min_matsubara_size(min_matsubara_size);

    const int max_matsubara_size = config.get_uint("/integration/max_matsubara_size", 128);
    matsubara_storage.set_max_matsubara_size(max_matsubara_size);
  }
} // namespace DiFfRG
