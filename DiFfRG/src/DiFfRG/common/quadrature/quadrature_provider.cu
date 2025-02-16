// DiFfRG
#include "DiFfRG/common/json.hh"
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

// external
#include <spdlog/spdlog.h>

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
        bool itp_found_T = (std::prev(it) != quadratures_d.end()) && (is_close(std::prev(it)->first, T, 1e-6 * T));

        if (it_found_T)
          return it;
        else if (itp_found_T)
          return std::prev(it);
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
        bool itp_found_T = (std::prev(it) != map.end()) && (is_close(std::prev(it)->first, E, 1e-6 * E));

        if (it_found_T)
          return it;
        else if (itp_found_T)
          return std::prev(it);
      }

      // create new entry in quadratures_d
      auto new_it = map.insert(std::make_pair(E, MatsubaraQuadrature<double>())).first;
      // Initialize the quadrature
      new_it->second.reinit(T_it->first, E);

      if (verbosity >= 0)
        spdlog::get("QuadratureProvider")
            ->info("Created MatsubaraQuadrature<double> with T = {:.4} and E = {:.4} and size = {}", T_it->first, E,
                   new_it->second.size());

      return new_it;
    }

    MatsubaraStorage::TemperatureIterator<float> MatsubaraStorage::find_T_f(const float T)
    {
      // search for T
      auto it = quadratures_f.lower_bound(T);

      if (it != quadratures_f.end()) {
        bool it_found_T = (is_close(it->first, T, 1e-6 * T));
        bool itp_found_T = (std::prev(it) != quadratures_f.end()) && (is_close(std::prev(it)->first, T, 1e-6 * T));

        if (it_found_T)
          return it;
        else if (itp_found_T)
          return std::prev(it);
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
        bool itp_found_T = (std::prev(it) != map.end()) && (is_close(std::prev(it)->first, E, 1e-6 * E));

        if (it_found_T)
          return it;
        else if (itp_found_T)
          return std::prev(it);
      }

      // create new entry in quadratures_f
      auto new_it = map.insert(std::make_pair(E, MatsubaraQuadrature<float>())).first;
      // Initialize the quadrature
      new_it->second.reinit(T_it->first, E);

      if (verbosity >= 0)
        spdlog::get("QuadratureProvider")
            ->info("Created MatsubaraQuadrature<float> with T = {:.4} and E = {:.4} and size = {}", T_it->first, E,
                   new_it->second.size());

      return new_it;
    }

    void MatsubaraStorage::set_verbosity(int v) { verbosity = v; }

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
        spdlog::get("QuadratureProvider")
            ->info("Created Quadrature<double> with order = {0} and type = {1}", order, (uint)type_it->first);

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
        spdlog::get("QuadratureProvider")
            ->info("Created Quadrature<float> with order = {0} and type = {1}", order, (uint)type_it->first);

      return new_it;
    }

    void QuadratureStorage::set_verbosity(int v) { verbosity = v; }
  } // namespace internal

  QuadratureProvider::QuadratureProvider() {}

  QuadratureProvider::QuadratureProvider(const JSONValue &json)
  {
    verbosity = json.get_int("/output/verbosity", 0);
    std::string folder = json.get_string("/output/folder", "./");
    std::string output_name = json.get_string("/output/name", "output");

    // create a spdlog logger for the quadrature provider
    build_logger("QuadratureProvider", folder + output_name + "_quadrature.log");

    if (verbosity >= 0) {
      spdlog::get("QuadratureProvider")->info("QuadratureProvider: Initialized quadrature provider.");
      matsubara_storage.set_verbosity(verbosity);
      quadrature_storage.set_verbosity(verbosity);
    }
  }
} // namespace DiFfRG