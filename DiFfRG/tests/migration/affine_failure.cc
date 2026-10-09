#include <DiFfRG/discretization/common/affine_constraint_metadata.hh>

struct LegacyModel {
  template <typename Constraints, typename BoundaryDofs, typename BoundaryPoints>
  void affine_constraints(Constraints &, const BoundaryDofs &, const BoundaryPoints &);
};

void probe()
{
  LegacyModel model;
  int constraints = 0;
  int context = 0;
  DiFfRG::internal::apply_model_affine_constraints(model, constraints, context);
}
