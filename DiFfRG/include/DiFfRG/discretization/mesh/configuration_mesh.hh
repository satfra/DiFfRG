#pragma once
#include "DiFfRG/common/math.hh"
#include <DiFfRG/common/utils.hh>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// fmt::format instead of std::format: libstdc++ ships <format> only from GCC
// 13, and spdlog's fmt is bundled anyway -- this keeps GCC 12 viable.
#include <spdlog/fmt/fmt.h>

namespace DiFfRG
{

  namespace Config
  {

    namespace internal
    {
      inline std::vector<std::string> string_to_substrings_array(const std::string str)
      {
        std::vector<std::string> array;
        std::istringstream ss(str);
        std::string buf;
        while (std::getline(ss, buf, ','))
          array.push_back(buf);
        return array;
      }
      inline std::array<double, 3> string_to_range(const std::string str)
      {
        std::vector<double> vec;
        std::istringstream ss(str);
        std::string buf;
        while (std::getline(ss, buf, ':'))
          vec.push_back(std::stod(buf));
        if (vec.size() != 3) throw std::runtime_error(str + " is not a range");

        std::array<double, 3> arr;
        std::copy_n(vec.begin(), 3, arr.begin());
        return arr;
      }
    } // namespace internal

    class GridAxis
    {
    public:
      GridAxis() = delete;
      GridAxis(double min_, double step_, double max_) : min(min_), max(max_), step(step_) { validate(); };

      GridAxis(std::string json_string)
      {
        auto subrange = internal::string_to_range(json_string);
        min = subrange[0];
        step = subrange[1];
        max = subrange[2];

        validate();
      };

      static std::string get_default_range() { return "0:1e-4:7e-3,7e-3:5e-4:9e-3,9e-3:1e-3:1.1e-2"; }

      double min;
      double max;
      double step;

      /**
       * Return a vector of uniform step widths covering [min, max].
       * If the configured step fits the interval (within is_close) it is used;
       * otherwise the count is increased by one and the width is adjusted.
       *
       * If @p origin_cell_centered is true, choose the smallest cell count whose adjusted width is less than the
       * configured width and which permits the first final cell to be centered on the origin after @p refinement
       * levels of global refinement. The range must start at zero in that case.
       */
      std::vector<double> get_stepwiths(const bool origin_cell_centered = false, const uint refinement = 0) const
      {
        if (origin_cell_centered) {
          if (!is_close(min, 0.0))
            throw std::invalid_argument(
                "Origin-centered rectangular meshes require every configured axis to start at zero.");

          const double refinement_factor = std::ldexp(1.0, refinement);
          auto steps = static_cast<uint>((max - min) / step);
          double local_step;
          do {
            ++steps;
            local_step = max / (static_cast<double>(steps) - 0.5 / refinement_factor);
          } while (!(local_step < step));

          return std::vector<double>(steps, local_step);
        }

        double local_step = step;
        auto steps = static_cast<uint>((max - min) / local_step);
        if (!is_close((min + local_step * steps), max)) {
          local_step = (max - min) / (steps + 1);
          ++steps;
        }
        return std::vector<double>(steps, local_step);
      }

    private:
      void validate()
      {
        bool condition = (min < max) && (0.0 < step) && (step <= (max - min));
        if (!condition) {
          throw std::runtime_error(fmt::format(
              "invalid range: min={}, step={}, max={} (require min < max and 0 < step <= max-min)", min, step, max));
        }
      }
    };
    namespace internal
    {
      inline void check_ranges_consistency(const std::vector<GridAxis> &ranges)
      {
        for (uint i = 1; i < ranges.size(); ++i)
          if (!is_close(ranges[i].min, ranges[i - 1].max))
            throw std::runtime_error("Your range definition is inconsistent!");
      }

      template <typename T, typename J> void append_range(T &to_append, const J &range)
      {
        for (const auto &element : range) {
          to_append.push_back(element);
        }
      }
    } // namespace internal

    template <int dim> class ConfigurationMesh
    {
    public:
      struct TriangulationData {
        dealii::Point<dim, double> lower_left;
        dealii::Point<dim, double> upper_right;
        std::vector<std::vector<double>> step_sizes = std::vector<std::vector<double>>(dim);
      };

      ConfigurationMesh() = delete;

      template <typename... Grids> ConfigurationMesh(unsigned int refinement, Grids... grids_args) : refine(refinement)
      {
        static_assert(sizeof...(Grids) == dim, "Number of grids must match dimension");
        static_assert((std::is_same_v<std::vector<GridAxis>, Grids> && ...),
                      "All grids must be of type std::vector<GridAxis>");
        grids = {grids_args...};
      }

      ConfigurationMesh(const DiFfRG::ConfigTree &config)
      {
        static_assert(dim >= 1 && dim <= 3, "ConfigurationMesh only supports dim = 1, 2, 3");

        const std::array<std::string, 3> grid_names = {"/discretization/grid/x_grid", "/discretization/grid/y_grid",
                                                       "/discretization/grid/z_grid"};

        for (int i = 0; i < dim; ++i) {
          auto subranges = internal::string_to_substrings_array(config.get_string_or_warn(grid_names[i], "0:0.1:1"));
          for (const auto &subrange_str : subranges)
            grids[i].emplace_back(subrange_str);
          internal::check_ranges_consistency(grids[i]);
        }

        refine = config.get_uint("/discretization/grid/refine", 0);
      }

      static std::string get_defaults()
      {
        std::string json_str = R"({
  "discretization": {
    "grid": {
      "x_grid": ")";
        json_str += GridAxis::get_default_range();
        json_str += R"(")";

        if constexpr (dim >= 2) {
          json_str += R"(,
      "y_grid": ")";
          json_str += GridAxis::get_default_range();
          json_str += R"(")";
        }

        if constexpr (dim >= 3) {
          json_str += R"(,
      "z_grid": ")";
          json_str += GridAxis::get_default_range();
          json_str += R"(")";
        }

        json_str += R"(,
      "refine": 0
    }
  }
})";
        return json_str;
      }

      /**
       * Return mutually consistent bounds and step widths for triangulation construction.
       *
       * If @p origin_cell_centered is true, the first final cell is centered on the origin along every axis.
       * All configured axes must start at zero in that case.
       */
      inline TriangulationData get_triangulation_data(const bool origin_cell_centered = false) const
      {
        TriangulationData data;
        for (int i = 0; i < dim; ++i) {
          data.lower_left[i] = grids[i].front().min;
          data.upper_right[i] = grids[i].back().max;
          for (std::size_t subrange = 0; subrange < grids[i].size(); ++subrange) {
            const bool center_first_subrange = origin_cell_centered && subrange == 0;
            internal::append_range(data.step_sizes[i], grids[i][subrange].get_stepwiths(center_first_subrange, refine));
          }
        }

        if (!origin_cell_centered) return data;

        const double refinement_factor = std::ldexp(1.0, refine);
        for (int axis = 0; axis < dim; ++axis) {
          data.lower_left[axis] = -data.step_sizes[axis].front() / (2.0 * refinement_factor);
        }

        return data;
      }

      inline std::vector<std::vector<double>> get_step_withs_for_triangulation() const
      {
        return get_triangulation_data().step_sizes;
      }

      inline dealii::Point<dim, double> get_lower_left() const { return get_triangulation_data().lower_left; }

      inline dealii::Point<dim, double> get_upper_right() const { return get_triangulation_data().upper_right; }

      std::array<std::vector<GridAxis>, dim> grids;
      uint refine;
    };

  } // namespace Config
} // namespace DiFfRG
