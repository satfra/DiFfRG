#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/variables/assembler/variables.hh>
#include <boilerplate/kt_models.hh>
#include <boilerplate/models.hh>

namespace
{
  using FVModel = DiFfRG::Testing::ModelBurgersKT<1>;
  using FVDiscretization = DiFfRG::FV::Discretization<FVModel, DiFfRG::RectangularMesh<1>>;
  using FVBaseAssembler = DiFfRG::FV::KurganovTadmor::Assembler<FVDiscretization, FVModel>;

  class FVApplicationAssembler : public FVBaseAssembler
  {
  public:
    using FVBaseAssembler::FVBaseAssembler;
  };

  using FEMModel = DiFfRG::Testing::ModelConstant<1>;
  using FEMDiscretization = DiFfRG::CG::Discretization<FEMModel, DiFfRG::RectangularMesh<1>>;
  using FEMBaseAssembler = DiFfRG::CG::Assembler<FEMDiscretization>;

  class FEMApplicationAssembler : public FEMBaseAssembler
  {
  public:
    using FEMBaseAssembler::FEMBaseAssembler;
  };

  struct VariablesModel {
    using Components = DiFfRG::ComponentDescriptor<DiFfRG::FEFunctionDescriptor<>>;

    void set_time(double) {}

    template <typename Residual, typename Solutions> void dt_variables(Residual &, const Solutions &) const {}

    template <int, typename Matrix, typename Solutions> void jacobian_variables(Matrix &, const Solutions &) const {}

    template <typename Helper, typename Output> void readouts_multiple(Helper &&, Output &) const {}
  };

  using VariablesBaseAssembler = DiFfRG::Variables::Assembler<VariablesModel>;

  class VariablesApplicationAssembler : public VariablesBaseAssembler
  {
  public:
    using VariablesBaseAssembler::VariablesBaseAssembler;
  };
} // namespace

void probe(FVDiscretization &fv_discretization, FVModel &fv_model, FEMDiscretization &fem_discretization,
           FEMModel &fem_model, VariablesModel &variables_model, const DiFfRG::ConfigTree &json)
{
  FVApplicationAssembler fv_inherited(fv_discretization, fv_model, json);
  FEMApplicationAssembler fem_inherited(fem_discretization, fem_model, json);
  VariablesApplicationAssembler variables_inherited(variables_model, json);
}
