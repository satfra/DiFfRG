#pragma once
#include <concepts>
#include <span>
namespace DiFfRG::FV::KurganovTadmor
{
  /// The hook runs after reconstruction and before evaluating model fluxes and sources.
  enum class AssemblyStage { residual, jacobian };

  /// Borrowed ranges of the actual mesh entities, valid until the mesh is rebuilt.
  template <typename Mesh> class AssemblyContextView
  {
  public:
    static constexpr int dim = Mesh::dimension;
    static constexpr size_t n_components = Mesh::components;
    using NumberType = typename Mesh::scalar_type;
    using FaceRange = std::span<const typename Mesh::Face>;
    using CellRange = std::span<const typename Mesh::Cell>;
    explicit AssemblyContextView(const Mesh &mesh) : faces_(mesh.faces()), cells_(mesh.cells()) {}
    const FaceRange &faces() const { return faces_; }
    const CellRange &cells() const { return cells_; }

  private:
    FaceRange faces_;
    CellRange cells_;
  };
  template <typename Context>
  concept HasAssemblyContextView = requires(const Context &context) {
    typename Context::NumberType;
    context.faces();
    context.cells();
  };
  template <typename Model, typename Context>
  concept HasFVKTAssemblyHook =
      HasAssemblyContextView<Context> && requires(Model &model, AssemblyStage stage, const Context &context) {
        { model.fv_kt_pre_assembly(stage, context) } -> std::same_as<void>;
      };
  template <typename Model, HasAssemblyContextView Context>
  void dispatch_fv_kt_pre_assembly(Model &model, AssemblyStage stage, const Context &context)
  {
    if constexpr (HasFVKTAssemblyHook<Model, Context>) model.fv_kt_pre_assembly(stage, context);
  }
} // namespace DiFfRG::FV::KurganovTadmor
