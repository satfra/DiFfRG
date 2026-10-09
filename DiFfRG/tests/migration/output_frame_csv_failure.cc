#include <DiFfRG/discretization/data/output_session.hh>

void probe(DiFfRG::OutputFrame<1, dealii::Vector<double>> &output)
{
  auto &table = output.csv("EoM.csv");
  (void)table;
}
