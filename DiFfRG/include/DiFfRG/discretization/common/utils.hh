#pragma once

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <tbb/parallel_for.h>

// standard library
#include <iterator>
#include <mutex>

namespace DiFfRG
{
  namespace internal
  {
    using namespace dealii;
    template <int dim, typename Cell>
    bool is_in_cell(const Cell &cell, const Point<dim> &point, const Mapping<dim> &mapping)
    {
      try {
        Point<dim> qp = mapping.transform_real_to_unit_cell(cell, point);
        if (GeometryInfo<dim>::is_inside_unit_cell(qp))
          return true;
        else
          return false;
      } catch (const typename Mapping<dim>::ExcTransformationFailed &) {
        // transformation failed, so assume the point is outside
        return false;
      }
    }

    template <int dim>
    dealii::Point<dim> get_origin(const dealii::DoFHandler<dim> &dof_handler,
                                  typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell)
    {
      using namespace dealii;
      using CellIterator = typename dealii::DoFHandler<1>::cell_iterator;

      auto l1_norm = [](const Point<dim> &p) {
        double norm = 0.;
        for (uint d = 0; d < dim; ++d)
          norm += std::abs(p[d]);
        return norm;
      };

      std::vector<Point<dim>> candidates;

      auto iterate_cell = [&](const CellIterator &cell) {
        std::array<Point<dim>, GeometryInfo<dim>::vertices_per_cell> vertices;
        for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i) {
          vertices[i] = cell->vertex(i);
        }
        // choose the one with the smallest l1 norm
        auto min_it = std::min_element(vertices.begin(), vertices.end(),
                                       [&](const auto &p1, const auto &p2) { return l1_norm(p1) < l1_norm(p2); });
        // add the point to the candidates
        candidates.push_back(*min_it);
      };

      for (const auto &cell : dof_handler.active_cell_iterators())
        iterate_cell(cell);

      // find the candidate with the smallest l1 norm
      auto min_it = std::min_element(candidates.begin(), candidates.end(),
                                     [&](const auto &p1, const auto &p2) { return l1_norm(p1) < l1_norm(p2); });

      Point<dim> origin = *min_it;
      EoM_cell = GridTools::find_active_cell_around_point(dof_handler, origin);

      return origin;
    }

  } // namespace internal

  /**
   * @brief Get the EoM point for a given solution and model in 1D. This is done by first checking the origin, and then
   * checking all cell borders in order to find a zero crossing. Then, the EoM point is found by bisection within the
   * cell.
   *
   * @tparam VectorType type of the solution vector.
   * @tparam Model type of the model.
   * @param EoM_cell the cell where the EoM point is located, will be set by the function. Is also used as a starting
   * point for the search.
   * @param sol the solution vector.
   * @param dof_handler a DoFHandler object associated with the solution vector.
   * @param mapping a Mapping object associated with the solution vector.
   * @param model numerical model providing a method EoM(const VectorType &)->double which we use to find a zero
   * crossing.
   * @param relative_tolerance the relative tolerance for the bisection method.
   * @return Point<dim> the point where the EoM is zero.
   */
  template <typename VectorType, typename EoMFUN, typename EoMPFUN>
  Point<1> get_EoM_point_1D(
      typename dealii::DoFHandler<1>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<1> &dof_handler, const dealii::Mapping<1> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, const auto &values) { return p; },
      const double relative_tolerance = 1e-5, const uint max_iter = 100)
  {
    constexpr uint dim = 1;

    using namespace dealii;
    using CellIterator = typename dealii::DoFHandler<1>::cell_iterator;
    auto EoM = Point<dim>();
    Vector<typename VectorType::value_type> values(dof_handler.get_fe().n_components());

    // We start by investigating the origin.
    const auto origin = internal::get_origin(dof_handler, EoM_cell);
    Functions::FEFieldFunction<dim, VectorType> fe_function(dof_handler, sol, mapping);
    fe_function.vector_value(origin, values);
    const double EoM_val = get_EoM(origin, values)[0];
    if (EoM_val >= 0.) {
      EoM = origin;
      return EoM;
    }

    auto check_cell = [&](const CellIterator &cell) {
      // Obtain the values at the vertices of the cell.
      std::array<double, GeometryInfo<dim>::vertices_per_cell> EoM_vals;
      std::array<Point<dim>, GeometryInfo<dim>::vertices_per_cell> vertices;
      // Vector<typename VectorType::value_type> values(dof_handler.get_fe().n_components());
      // Functions::FEFieldFunction<dim, VectorType> fe_function(dof_handler, sol, mapping);

      fe_function.set_active_cell(cell);
      for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i) {
        vertices[i] = cell->vertex(i);
        fe_function.vector_value(vertices[i], values);
        EoM_vals[i] = get_EoM(vertices[i], values)[0];
      }

      bool cell_has_EoM = false;
      // Find if the cell has an EoM point, i.e. if the values at some vertices have different signs.
      for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
        for (uint j = 0; j < i; ++j)
          if (EoM_vals[i] * EoM_vals[j] < 0.) {
            cell_has_EoM = true;
            i = GeometryInfo<dim>::vertices_per_cell;
            break;
          }

      return cell_has_EoM;
    };

    auto find_EoM = [&](const CellIterator &cell, CellIterator &m_EoM_cell, Point<dim> &m_EoM,
                        double &EoM_val) -> bool {
      std::array<Point<dim>, GeometryInfo<dim>::vertices_per_cell> vertices;
      for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
        vertices[i] = cell->vertex(i);
      fe_function.set_active_cell(cell);

      auto p1 = vertices[0];
      auto p2 = vertices[1];
      if (p2[0] < p1[0]) std::swap(p1, p2);
      auto p = m_EoM[0] <= p2[0] && m_EoM[0] >= p1[0] ? m_EoM : (p1 + p2) / 2.;

      EoM_val = get_EoM(p, values)[0];
      double err = std::abs(EoM_val);
      uint iter = 0;

      while (err > relative_tolerance) {
        if (EoM_val < 0.)
          p1 = p;
        else
          p2 = p;
        p = (p1 + p2) / 2.;

        fe_function.vector_value(p, values);
        EoM_val = get_EoM(p, values)[0];
        err = std::abs(EoM_val);

        if (iter > max_iter) {
          m_EoM = EoM_postprocess(p, values);
          m_EoM_cell = cell;
          return false;
        }
        iter++;
      }

      m_EoM = EoM_postprocess(p, values);
      m_EoM_cell = cell;

      return true;
    };

    std::vector<typename dealii::DoFHandler<dim>::cell_iterator> cell_candidates;
    std::vector<double> value_candidates;
    std::vector<Point<dim>> EoM_candidates;

    CellIterator t_EoM_cell;
    Point<dim> t_EoM;
    double t_EoM_value = 0.;

    // Check the cell from the previous iteration first
    if (EoM_cell != dof_handler.active_cell_iterators().end())
      if (check_cell(EoM_cell)) {
        if (find_EoM(EoM_cell, t_EoM_cell, t_EoM, t_EoM_value)) {
          EoM_cell = t_EoM_cell;
          EoM = t_EoM;
          return EoM;
        }
        cell_candidates.push_back(t_EoM_cell);
        value_candidates.push_back(std::abs(t_EoM_value));
        EoM_candidates.push_back(t_EoM);
      }

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (check_cell(cell)) {
        if (find_EoM(cell, t_EoM_cell, t_EoM, t_EoM_value)) {
          EoM_cell = t_EoM_cell;
          EoM = t_EoM;
          return EoM;
        }
        cell_candidates.push_back(t_EoM_cell);
        value_candidates.push_back(std::abs(t_EoM_value));
        EoM_candidates.push_back(t_EoM);
      }

    if (cell_candidates.size() == 0) {
      EoM_cell = GridTools::find_active_cell_around_point(dof_handler, origin);
      return origin;
    } else if (cell_candidates.size() > 1) {
      // If we have more than one candidate, we choose the one with the smallest EoM value.
      auto min_it = std::min_element(value_candidates.begin(), value_candidates.end());
      auto min_idx = std::distance(value_candidates.begin(), min_it);
      EoM_cell = cell_candidates[min_idx];
      EoM = EoM_candidates[min_idx];
      return EoM;
    }

    find_EoM(cell_candidates[0], EoM_cell, EoM, t_EoM_value);

    return EoM;
  }

  /**
   * @brief Get the EoM point for a given solution and model.
   *
   * @tparam dim dimension of the problem.
   * @tparam VectorType type of the solution vector.
   * @tparam Model type of the model.
   * @param EoM_cell the cell where the EoM point is located, will be set by the function. Is also used as a starting
   * point for the search.
   * @param sol the solution vector.
   * @param dof_handler a DoFHandler object associated with the solution vector.
   * @param mapping a Mapping object associated with the solution vector.
   * @param model numerical model providing a method EoM(const VectorType &)->double which we use to find a zero
   * crossing.
   * @param relative_tolerance the relative tolerance for the bisection method.
   * @return Point<dim> the point where the EoM is zero.
   */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  Point<dim> get_EoM_point(
      typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, const auto &values) { return p; },
      const double relative_tolerance = 1e-5, const uint max_iter = 100)
  {
    if (max_iter == 0) return internal::get_origin(dof_handler, EoM_cell);

    if constexpr (dim == 1) {
      return get_EoM_point_1D(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess, relative_tolerance,
                              max_iter);
    } else if constexpr (dim > 1) {
      throw std::runtime_error("get_EoM_point is not yet implemented for dim > 1. Please set EoM_max_iter = 0.");
    }
  }
} // namespace DiFfRG
