#pragma once

namespace DiFfRG
{
  namespace internal
  {
    template <typename> inline constexpr bool data_output_removed = false;
  }

  /// Removed; use OutputSession. Naming it gives a compile error explaining the replacement.
  template <unsigned int dim, typename VectorType>
  class [[deprecated("DataOutput was removed. Construct OutputPath, then OutputSession<Assembler>, and pass the "
                     "session to the timestepper instead. For assembler callbacks, override "
                     "attach_data_output(OutputFrame<dim, VectorType>&, ...) instead. Inside the callback, replace "
                     "data_out.fe_output().attach(...) with auto fields = output.fields(); fields.attach(...)")]]
  DataOutput
  {
    static_assert(internal::data_output_removed<VectorType>,
                  "DataOutput was removed. Construct OutputPath, then OutputSession<Assembler>, and pass the "
                  "session to the timestepper instead. For assembler callbacks, override "
                  "attach_data_output(OutputFrame<dim, VectorType>&, ...) instead. Inside the callback, replace "
                  "data_out.fe_output().attach(...) with auto fields = output.fields(); fields.attach(...)");
  };
} // namespace DiFfRG
