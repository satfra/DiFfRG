#pragma once

// DiFfRG
#include <DiFfRG/model/component_descriptor.hh>

// external libraries
#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// standard library
#include <utility>
#include <vector>

namespace DiFfRG
{
  /// A set of dofs of one FE function and their support points: points[i] belongs to dofs.nth_index_in_set(i).
  template <int dim> struct AffineConstraintComponentView {
    /// Global dof indices.
    const dealii::IndexSet &dofs;
    /// Support point of each dof, in the order of @p dofs.
    const std::vector<dealii::Point<dim>> &points;
  };

  /**
   * @brief What a model's affine-constraint hooks get to see: the dofs and support points of each FE function.
   *
   * Passed as `context` to `apply_boundary_affine_constraints(constraints, context)` and
   * `apply_affine_constraints(constraints, context)` of the model (see def::ConstrainOriginBoundaryPointToZero for an
   * example). A model that fixes \f$u(x_0) = c\f$ for its function "u" picks a dof from
   * `context.template support<"u">()` (or `boundary<"u">()`) and adds the line to the deal.II AffineConstraints.
   *
   * @tparam Components_ The model's ComponentDescriptor; FE functions are addressed by their names.
   * @tparam dim Spatial dimension.
   */
  template <typename Components_, int dim> class AffineConstraintContext
  {
  public:
    using Components = Components_;
    static constexpr int dimension = dim;

    /// Built by the assembler; one entry per FE function in each vector.
    AffineConstraintContext(const std::vector<dealii::IndexSet> &component_boundary_dofs,
                            const std::vector<std::vector<dealii::Point<dim>>> &component_boundary_points,
                            const std::vector<dealii::IndexSet> &component_support_dofs,
                            const std::vector<std::vector<dealii::Point<dim>>> &component_support_points)
        : component_boundary_dofs(component_boundary_dofs), component_boundary_points(component_boundary_points),
          component_support_dofs(component_support_dofs), component_support_points(component_support_points)
    {
    }

    /// Built by the assembler from internal::AffineConstraintMetadata.
    template <typename Metadata>
    explicit AffineConstraintContext(const Metadata &metadata)
        : AffineConstraintContext(metadata.component_boundary_dofs, metadata.component_boundary_points,
                                  metadata.component_support_dofs, metadata.component_support_points)
    {
    }

    /// Index of the FE function @p component_name in the ComponentDescriptor.
    template <FixedString component_name> static consteval unsigned int component_index()
    {
      return static_cast<unsigned int>(typename Components::FEFunction_Descriptor{}(component_name));
    }

    /// Declared size of the FE function @p component_name in the ComponentDescriptor.
    template <FixedString component_name> static consteval std::size_t component_size()
    {
      return Components::FEFunction_Descriptor::sizes[component_index<component_name>()];
    }

    /// The dofs of @p component_name on the domain boundary, with their support points.
    template <FixedString component_name> AffineConstraintComponentView<dim> boundary() const
    {
      constexpr auto c = component_index<component_name>();
      return {component_boundary_dofs[c], component_boundary_points[c]};
    }

    /// All dofs of @p component_name, with their support points.
    template <FixedString component_name> AffineConstraintComponentView<dim> support() const
    {
      constexpr auto c = component_index<component_name>();
      return {component_support_dofs[c], component_support_points[c]};
    }

  private:
    const std::vector<dealii::IndexSet> &component_boundary_dofs;
    const std::vector<std::vector<dealii::Point<dim>>> &component_boundary_points;
    const std::vector<dealii::IndexSet> &component_support_dofs;
    const std::vector<std::vector<dealii::Point<dim>>> &component_support_points;
  };

  namespace internal
  {
    template <typename> inline constexpr bool accepts_affine_constraint_context = false;

    /// Call the model's affine_constraints(constraints, context).
    template <typename Model, typename Constraints, typename Context>
    void apply_model_affine_constraints(Model &model, Constraints &constraints, const Context &context)
    {
      if constexpr (requires { model.affine_constraints(constraints, context); }) {
        model.affine_constraints(constraints, context);
      } else {
        static_assert(accepts_affine_constraint_context<Model>,
                      "The three-argument affine_constraints callback was removed. Implement "
                      "apply_boundary_affine_constraints(constraints, context) or "
                      "apply_affine_constraints(constraints, context) using AffineConstraintContext instead");
      }
    }

    /// Storage behind AffineConstraintContext.
    template <int dim> struct AffineConstraintMetadata {
      std::vector<dealii::IndexSet> component_boundary_dofs;
      std::vector<std::vector<dealii::Point<dim>>> component_boundary_points;
      std::vector<dealii::IndexSet> component_support_dofs;
      std::vector<std::vector<dealii::Point<dim>>> component_support_points;
    };

    /// Collect the dofs and support points of every FE function from the discretization.
    template <typename Components, int dim, typename Discretization>
    AffineConstraintMetadata<dim> build_affine_constraint_metadata(const Discretization &discretization)
    {
      const auto &dof_handler = discretization.get_dof_handler();
      AffineConstraintMetadata<dim> metadata;
      metadata.component_boundary_dofs.resize(Components::count_fe_functions());
      metadata.component_boundary_points.resize(Components::count_fe_functions());
      metadata.component_support_dofs.resize(Components::count_fe_functions());
      metadata.component_support_points.resize(Components::count_fe_functions());

      for (unsigned int c = 0; c < Components::count_fe_functions(); ++c) {
        dealii::ComponentMask component_mask(Components::count_fe_functions(), false);
        component_mask.set(c, true);

        metadata.component_boundary_dofs[c] = dealii::DoFTools::extract_boundary_dofs(dof_handler, component_mask);
        metadata.component_boundary_points[c].resize(metadata.component_boundary_dofs[c].n_elements());
        for (unsigned int i = 0; i < metadata.component_boundary_dofs[c].n_elements(); ++i)
          metadata.component_boundary_points[c][i] =
              discretization.get_support_point(metadata.component_boundary_dofs[c].nth_index_in_set(i));

        metadata.component_support_dofs[c] = dealii::DoFTools::extract_dofs(dof_handler, component_mask);
        metadata.component_support_points[c].resize(metadata.component_support_dofs[c].n_elements());
        for (unsigned int i = 0; i < metadata.component_support_dofs[c].n_elements(); ++i)
          metadata.component_support_points[c][i] =
              discretization.get_support_point(metadata.component_support_dofs[c].nth_index_in_set(i));
      }

      return metadata;
    }
  } // namespace internal
} // namespace DiFfRG
