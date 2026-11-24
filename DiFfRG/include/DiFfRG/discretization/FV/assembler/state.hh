#pragma once
#include <DiFfRG/discretization/FV/assembler/reconstruction_tools.hh>
#include <DiFfRG/discretization/FV/mesh.hh>
#include <deal.II/lac/full_matrix.h>
#include <memory>
namespace DiFfRG::FV::KurganovTadmor
{
  struct JacobianDependencies {
    std::vector<dealii::types::global_dof_index> to_dofs, from_dofs;
  };
  template <typename NumberType, size_t n_components> struct CellState {
    using scalar_type = NumberType;
    static constexpr size_t components = n_components;
    std::array<NumberType, n_components> u{};
    JacobianDependencies source_dependencies;
  };
  template <int dim, typename NumberType, size_t n_components> struct FaceState {
    FV::FaceReconstructionState<dim, NumberType, n_components> reconstruction{};
    bool reconstruction_valid = false;
    JacobianDependencies dependencies;
    std::unique_ptr<internal::BoundaryReconstructionStencilTopologyData<dim, n_components>> boundary;
    int batch_position = -1;
    std::array<NumberType, n_components> residual{};
    dealii::FullMatrix<NumberType> jacobian;
  };
  template <int dim, typename NumberType, size_t n_components>
  using Mesh = FV::Mesh<dim, CellState<NumberType, n_components>, FaceState<dim, NumberType, n_components>>;
} // namespace DiFfRG::FV::KurganovTadmor
