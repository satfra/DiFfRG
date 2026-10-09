#include <DiFfRG/discretization/data/output_session.hh>

void probe(DiFfRG::OutputFrame<1, dealii::Vector<double>> &output) { output.set_Lambda(1.0); }
