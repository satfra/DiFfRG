#pragma once
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/numbers.h>
#include <array>
#include <functional>
#include <span>
#include <vector>

namespace DiFfRG::FV
{
  template <typename Discretization> class DealIIBackend;

  /// Complete FV entities. Payloads provide scalar_type, components and the scheme's numerical state.
  /// Construction finishes before connectivity is wired. References remain valid until the next rebuild.
  template <int dim, typename CellState, typename FaceState> class Mesh
  {
  public:
    static constexpr int dimension = dim;
    static constexpr size_t components = CellState::components;
    using scalar_type = typename CellState::scalar_type;
    struct Face;
    struct Cell {
      CellState state{};
      using scalar_type = typename CellState::scalar_type;
      static constexpr size_t components = CellState::components;
      static constexpr int dimension = dim;
      static constexpr size_t n_faces = 2 * dim;
      Cell() { dof_indices.fill(dealii::numbers::invalid_dof_index); }
      Cell(const Cell &) = delete;
      Cell &operator=(const Cell &) = delete;
      Cell(Cell &&) = default;
      Cell &operator=(Cell &&) = default;
      unsigned int index = 0;
      bool is_boundary_ghost = false;
      dealii::Point<dim> x;
      std::array<dealii::types::global_dof_index, components> dof_indices;
      std::vector<dealii::Point<dim>> quadrature_points;
      std::vector<scalar_type> jxw;
      std::vector<std::reference_wrapper<const Face>> faces;
      std::vector<std::reference_wrapper<const Cell>> neighbors;
      const auto &center() const { return x; }
      const auto &values() const { return state.u; }
      const auto &dofs() const { return dof_indices; }
      const auto &neighbor_center(unsigned int f) const { return neighbors[f].get().center(); }
      const auto &neighbor_values(unsigned int f) const { return neighbors[f].get().values(); }
      const auto &neighbor_dofs(unsigned int f) const { return neighbors[f].get().dofs(); }
      const auto &face_center(unsigned int f) const { return faces[f].get().center; }
      auto boundary_id(unsigned int f) const { return faces[f].get().boundary_id; }
    };
    struct Face {
      FaceState state;
      Face(Cell &minus, Cell &plus, unsigned int mf, unsigned int pf, const dealii::Point<dim> &center,
           const dealii::Tensor<1, dim> &normal, double jxw, dealii::types::boundary_id boundary_id)
          : minus(minus), plus(plus), minus_face_index(mf), plus_face_index(pf), center(center), normal(normal),
            jxw(jxw), boundary_id(boundary_id)
      {
      }
      Face(const Face &) = delete;
      Face &operator=(const Face &) = delete;
      Face(Face &&) = default;
      Face &operator=(Face &&) = delete;
      const Cell &minus;
      const Cell &plus;
      unsigned int minus_face_index;
      unsigned int plus_face_index;
      dealii::Point<dim> center;
      dealii::Tensor<1, dim> normal;
      double jxw;
      dealii::types::boundary_id boundary_id;
      bool at_boundary() const { return plus.is_boundary_ghost; }
    };
    Mesh() = default;
    Mesh(const Mesh &) = delete;
    Mesh &operator=(const Mesh &) = delete;
    Mesh(Mesh &&) = delete;
    Mesh &operator=(Mesh &&) = delete;
    std::span<Cell> cells() { return cells_; }
    std::span<const Cell> cells() const { return cells_; }
    std::span<Cell> ghost_cells() { return ghost_cells_; }
    std::span<const Cell> ghost_cells() const { return ghost_cells_; }
    std::span<Face> faces() { return faces_; }
    std::span<const Face> faces() const { return faces_; }
    const Face &face(unsigned int cell, unsigned int f) const { return cells_[cell].faces[f].get(); }

  private:
    template <typename Discretization> friend class DealIIBackend;
    std::vector<Cell> cells_;
    std::vector<Cell> ghost_cells_;
    std::vector<Face> faces_;
  };
} // namespace DiFfRG::FV
