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
      /// Split @p str at commas.
      inline std::vector<std::string> string_to_substrings_array(const std::string str)
      {
        std::vector<std::string> array;
        std::istringstream ss(str);
        std::string buf;
        while (std::getline(ss, buf, ','))
          array.push_back(buf);
        return array;
      }
      /// Parse "min:step:max"; throws if there are not exactly three numbers.
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

    /**
     * @brief One subrange [min, max] of a grid axis, divided into cells of width at most step.
     *
     * Requires min < max and 0 < step ≤ max - min (throws otherwise).
     */
    class GridAxis
    {
    public:
      GridAxis() = delete;
      /// From the three numbers.
      GridAxis(double min_, double step_, double max_) : min(min_), max(max_), step(step_) { validate(); };

      /// From a string "min:step:max", e.g. "0:0.1:1".
      GridAxis(std::string json_string)
      {
        auto subrange = internal::string_to_range(json_string);
        min = subrange[0];
        step = subrange[1];
        max = subrange[2];

        validate();
      };

      /// The axis string advertised by ConfigurationMesh::get_defaults().
      static std::string get_default_range() { return "0:1e-4:7e-3,7e-3:5e-4:9e-3,9e-3:1e-3:1.1e-2"; }

      /// Lower end.
      double min;
      /// Upper end.
      double max;
      /// Maximal cell width.
      double step;

      [[deprecated("Use get_step_widths")]] std::vector<double> get_stepwiths(const bool origin_cell_centered = false,
                                                                          const uint refinement = 0) const
      {
        return get_step_widths(origin_cell_centered, refinement);
      }

      /**
       * @brief The cell widths of this subrange: \f$n\f$ equal widths \f$h\f$.
       *
       * Normally \f$n = \lfloor (max - min)/step \rfloor\f$ and \f$h = step\f$ if \f$min + n\,step = max\f$
       * (up to rounding); otherwise \f$n\f$ is increased by one and \f$h = (max - min)/n < step\f$.
       *
       * If @p origin_cell_centered (requires min = 0, throws otherwise): the smallest \f$n\f$ with
       * \f$h = max / (n - 2^{-r-1}) < step\f$, \f$r\f$ = @p refinement. The cells then cover
       * \f$[-h\,2^{-r-1}, max]\f$, so that after \f$r\f$ global refinements the first cell is centered on 0.
       */
      std::vector<double> get_step_widths(const bool origin_cell_centered = false, const uint refinement = 0) const
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
      /// Throw unless each subrange starts where the previous one ends.
      inline void check_ranges_consistency(const std::vector<GridAxis> &ranges)
      {
        for (uint i = 1; i < ranges.size(); ++i)
          if (!is_close(ranges[i].min, ranges[i - 1].max))
            throw std::runtime_error("Your range definition is inconsistent!");
      }

      /// Append all elements of @p range to @p to_append.
      template <typename T, typename J> void append_range(T &to_append, const J &range)
      {
        for (const auto &element : range) {
          to_append.push_back(element);
        }
      }
    } // namespace internal

    // clang-format off
    /**
     * @brief A rectangular grid in field space, each axis a sequence of subranges with their own cell width.
     *
     * Each axis is a string of comma-separated subranges `min:step:max`, which must be contiguous (each `min` equals
     * the previous `max`). Each subrange is divided into equal cells of width at most `step`, see
     * GridAxis::get_step_widths. Example: `"0:0.01:0.1,0.1:0.05:1"`.
     *
     * | Key                                  | Type   | Default (if missing) | Meaning |
     * |--------------------------------------|--------|----------------------|---------|
     * | `/discretization/grid/x_grid`        | string | "0:0.1:1" (warns)    | first axis |
     * | `/discretization/grid/y_grid`        | string | "0:0.1:1" (warns)    | second axis (dim ≥ 2) |
     * | `/discretization/grid/z_grid`        | string | "0:0.1:1" (warns)    | third axis (dim = 3) |
     * | `/discretization/grid/refine`        | uint   | 0                    | number of global refinements of the mesh (each halves all cells) |
     *
     * get_defaults() writes a different axis string (GridAxis::get_default_range()).
     *
     * @tparam dim Field-space dimension, 1 to 3.
     */
    // clang-format on
    template <int dim> class ConfigurationMesh
    {
    public:
      /// What a mesh needs to build the coarse grid.
      struct TriangulationData {
        /// Lower corner of the domain.
        dealii::Point<dim, double> lower_left;
        /// Upper corner of the domain.
        dealii::Point<dim, double> upper_right;
        /// Per axis, the widths of the coarse cells in order.
        std::vector<std::vector<double>> step_sizes = std::vector<std::vector<double>>(dim);
      };

      ConfigurationMesh() = delete;

      /// From explicit axes: @p refinement and one std::vector<GridAxis> per dimension.
      template <typename... Grids> ConfigurationMesh(unsigned int refinement, Grids... grids_args) : refine(refinement)
      {
        static_assert(sizeof...(Grids) == dim, "Number of grids must match dimension");
        static_assert((std::is_same_v<std::vector<GridAxis>, Grids> && ...),
                      "All grids must be of type std::vector<GridAxis>");
        grids = {grids_args...};
      }

      /// Read the keys of the table above; throws if a subrange is invalid or the subranges are not contiguous.
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

      /// A JSON parameter-file fragment with all keys (axes set to GridAxis::get_default_range()).
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
       * @brief Domain corners and coarse cell widths.
       *
       * If @p origin_cell_centered (all axes must start at 0): the first cell of every axis is chosen such that after
       * `refine` refinements the first cell is centered on 0, and lower_left is shifted below 0 accordingly; see
       * GridAxis::get_step_widths.
       */
      inline TriangulationData get_triangulation_data(const bool origin_cell_centered = false) const
      {
        TriangulationData data;
        for (int i = 0; i < dim; ++i) {
          data.lower_left[i] = grids[i].front().min;
          data.upper_right[i] = grids[i].back().max;
          for (std::size_t subrange = 0; subrange < grids[i].size(); ++subrange) {
            const bool center_first_subrange = origin_cell_centered && subrange == 0;
            internal::append_range(data.step_sizes[i], grids[i][subrange].get_step_widths(center_first_subrange, refine));
          }
        }

        if (!origin_cell_centered) return data;

        const double refinement_factor = std::ldexp(1.0, refine);
        for (int axis = 0; axis < dim; ++axis) {
          data.lower_left[axis] = -data.step_sizes[axis].front() / (2.0 * refinement_factor);
        }

        return data;
      }

      /// The coarse cell widths per axis (not origin-centered).
      inline std::vector<std::vector<double>> get_step_widths_for_triangulation() const
      {
        return get_triangulation_data().step_sizes;
      }
      [[deprecated("Use get_step_widths_for_triangulation")]] std::vector<std::vector<double>>
      get_step_withs_for_triangulation() const
      {
        return get_step_widths_for_triangulation();
      }

      /// Lower corner of the domain (not origin-centered).
      inline dealii::Point<dim, double> get_lower_left() const { return get_triangulation_data().lower_left; }

      /// Upper corner of the domain.
      inline dealii::Point<dim, double> get_upper_right() const { return get_triangulation_data().upper_right; }

      /// The subranges of each axis.
      std::array<std::vector<GridAxis>, dim> grids;
      /// Number of global refinements (`/discretization/grid/refine`).
      uint refine;
    };

  } // namespace Config
} // namespace DiFfRG
