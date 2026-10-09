#include <catch2/catch_test_macros.hpp>

#include <DiFfRG/common/stream_filter.hh>

#include <iostream>
#include <sstream>

using namespace DiFfRG;

namespace
{
  /// The prefix Init() uses to drop Kokkos' architecture warning.
  constexpr std::string_view kokkos_arch_warning =
      "Kokkos::Cuda::initialize WARNING: running kernels compiled for compute capability";

  /// Feed @p input through the filter one character at a time and return what reached the sink.
  std::string filtered(std::string_view input, std::string_view prefix = kokkos_arch_warning)
  {
    std::ostringstream sink;
    LinePrefixFilter filter(sink.rdbuf(), prefix);
    std::ostream out(&filter);
    for (const char c : input)
      out.put(c);
    out.flush();
    return sink.str();
  }
} // namespace

TEST_CASE("The architecture warning is dropped", "[stream_filter]")
{
  // Verbatim from Kokkos_Cuda_Instance.cpp.
  const std::string warning = "Kokkos::Cuda::initialize WARNING: running kernels compiled for compute capability 7.5 "
                              "on device with compute capability 8.9 , this will likely reduce potential "
                              "performance.\n";
  CHECK(filtered(warning).empty());
  CHECK(filtered("before\n" + warning + "after\n") == "before\nafter\n");
}

// The whole reason the filter matches incrementally rather than buffering lines: Kokkos::initialize
// calls Kokkos::abort() when its architecture is *above* the device, and the process dies inside
// the filtered scope. A message that were withheld waiting for a newline would be lost, and the run
// would die with no explanation at all.
TEST_CASE("Nothing else is withheld", "[stream_filter]")
{
  SECTION("the sibling abort message passes")
  {
    const std::string error = "Kokkos::Cuda::initialize ERROR: running kernels compiled for compute capability 9.0 on "
                              "device with compute capability 8.9 is not supported by CUDA!\n";
    CHECK(filtered(error) == error);
  }

  SECTION("a diverged line reaches the sink before its newline")
  {
    // The abort case as it actually happens: Kokkos writes its message and the process dies inside
    // the filtered scope, so nothing is ever flushed and no newline may have been written yet. The
    // bytes have to be through the filter already. This is the one check that separates the
    // incremental filter from a line-buffering one -- do not let it be flushed first.
    std::ostringstream sink;
    LinePrefixFilter filter(sink.rdbuf(), kokkos_arch_warning);
    std::ostream out(&filter);

    const std::string dying = "Kokkos::Cuda::initialize ERROR: about to abort";
    out << dying;
    CHECK(sink.str() == dying);
  }

  SECTION("a line sharing a long prefix passes in full")
  {
    // Diverges only at the last character of the pattern.
    const std::string other = "Kokkos::Cuda::initialize WARNING: running kernels compiled for compute capabilities "
                              "that do not exist\n";
    CHECK(filtered(other) == other);
  }

  SECTION("an unterminated line is released on flush")
  {
    CHECK(filtered("Kokkos::Cuda::ini") == "Kokkos::Cuda::ini");
    CHECK(filtered("unrelated, no newline") == "unrelated, no newline");
  }

  SECTION("the matching text is dropped only at the start of a line")
  {
    const std::string embedded = std::string("note: ") + std::string(kokkos_arch_warning) + " 7.5\n";
    CHECK(filtered(embedded) == embedded);
  }

  SECTION("unrelated output is untouched")
  {
    const std::string other = "Kokkos::Cuda::initialize WARNING: something else entirely\nplain line\n\n";
    CHECK(filtered(other) == other);
  }
}

TEST_CASE("The filter is removed when its scope ends", "[stream_filter]")
{
  std::ostringstream capture;
  std::streambuf *const original = std::cerr.rdbuf(capture.rdbuf());
  {
    ScopedLineFilter guard(std::cerr, "drop me");
    std::cerr << "drop me, this is the warning\n" << "keep me\n" << std::flush;
  }
  std::cerr << "after the scope\n" << std::flush;
  std::cerr.rdbuf(original);

  CHECK(capture.str() == "keep me\nafter the scope\n");
}
