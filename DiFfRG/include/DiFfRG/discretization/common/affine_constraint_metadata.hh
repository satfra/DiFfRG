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
  template <int dim> struct AffineConstraintComponentView {
    const dealii::IndexSet &dofs;
    const std::vector<dealii::Point<dim>> &points;
  };

  template <typename Components_, int dim> class AffineConstraintContext
  {
  public:
    using Components = Components_;
    static constexpr int dimension = dim;

    AffineConstraintContext(const std::vector<dealii::IndexSet> &component_boundary_dofs,
                            const std::vector<std::vector<dealii::Point<dim>>> &component_boundary_points,
                            const std::vector<dealii::IndexSet> &component_support_dofs,
                            const std::vector<std::vector<dealii::Point<dim>>> &component_support_points)
        : component_boundary_dofs(component_boundary_dofs), component_boundary_points(component_boundary_points),
          component_support_dofs(component_support_dofs), component_support_points(component_support_points)
    {
    }

    template <typename Metadata>
    explicit AffineConstraintContext(const Metadata &metadata)
        : AffineConstraintContext(metadata.component_boundary_dofs, metadata.component_boundary_points,
                                  metadata.component_support_dofs, metadata.component_support_points)
    {
    }

    template <FixedString component_name> static consteval unsigned int component_index()
    {
      return static_cast<unsigned int>(typename Components::FEFunction_Descriptor{}(component_name));
    }

    template <FixedString component_name> static consteval std::size_t component_size()
    {
      return Components::FEFunction_Descriptor::sizes[component_index<component_name>()];
    }

    template <FixedString component_name> AffineConstraintComponentView<dim> boundary() const
    {
      constexpr auto c = component_index<component_name>();
      return {component_boundary_dofs[c], component_boundary_points[c]};
    }

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

    template <int dim> struct AffineConstraintMetadata {
      std::vector<dealii::IndexSet> component_boundary_dofs;
      std::vector<std::vector<dealii::Point<dim>>> component_boundary_points;
      std::vector<dealii::IndexSet> component_support_dofs;
      std::vector<std::vector<dealii::Point<dim>>> component_support_points;
    };

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
