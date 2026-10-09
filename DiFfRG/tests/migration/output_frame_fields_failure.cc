#include <DiFfRG/discretization/data/output_session.hh>

void probe(DiFfRG::OutputFrame<1, dealii::Vector<double>> &output)
{
  // Bind a reference: fe_output() returns FEOutput& and `auto fields = ...` would COPY it.
  // FEOutput has no usable copy constructor, so the copy fails first and the compiler never
  // instantiates fe_output() -- meaning its removal static_assert (the message this contract
  // asserts) is never emitted.
  auto &fields = output.fe_output();
  (void)fields;
}
