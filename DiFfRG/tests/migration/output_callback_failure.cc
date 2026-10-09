#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/data_output.hh>

using Vector = dealii::Vector<double>;
using Matrix = dealii::SparseMatrix<double>;
using Assembler = DiFfRG::AbstractAssembler<Vector, Matrix, 1>;
using LegacyOutput = DiFfRG::DataOutput<1, Vector>;
using LegacyCallback = void (Assembler::*)(LegacyOutput &, const Vector &, const Vector &, const Vector &,
                                           const Vector &);

LegacyCallback callback = static_cast<LegacyCallback>(&Assembler::attach_data_output);
