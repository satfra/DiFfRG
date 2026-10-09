#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/discretization/data/hdf5_frame.hh>
#include <DiFfRG/discretization/data/hdf5_writer.hh>

#include <unistd.h>

#include <filesystem>
#include <string>
#include <vector>

//--------------------------------------------------------------------------------------------
// HDF5FrameWriter in isolation: no Kokkos, no deal.II, no GPU -- just frames of actions and the
// thread that runs them. Everything asserted here is a property the rest of the output stack
// relies on but never exercises: what happens when a write fails, that a full queue blocks
// rather than drops, and that the file is complete and readable once drained.
//--------------------------------------------------------------------------------------------

namespace
{
  using namespace DiFfRG;

  std::filesystem::path make_file(const std::string &stem)
  {
    const auto path = std::filesystem::temp_directory_path() /
                      ("diffrg-writer-test-" + stem + "-" + std::to_string(::getpid()) + ".h5");
    std::filesystem::remove(path);
    auto file = DiFfRG::hdf5::File::open(path.string(), DiFfRG::hdf5::Access::Truncate);
    file.root().create_group("frames");
    file.flush();
    file.close();
    return path;
  }

  /** A frame that writes one 1-element dataset named after its index into /frames. */
  HDF5Frame counting_frame(const std::string &path, const unsigned int index)
  {
    HDF5Frame frame;
    frame.path = path;
    frame.time = 0.5 * index;
    frame.actions.emplace_back([index](HDF5FrameContext &context) {
      auto space = DiFfRG::hdf5::Dataspace::simple({1});
      auto dataset =
          context.group("frames").create_dataset(std::to_string(index), DiFfRG::hdf5::type_of<double>(), space);
      const double value = static_cast<double>(index);
      dataset.write(&value, 1);
    });
    return frame;
  }

  std::size_t count_frames(const std::filesystem::path &path)
  {
    auto file = DiFfRG::hdf5::File::open(path.string(), DiFfRG::hdf5::Access::ReadOnly);
    auto frames = file.root().open_group("frames");
    return frames.child_names().size();
  }
} // namespace

TEST_CASE("Queued frames all reach disk, in order", "[output][hdf5][writer][thread-safe]")
{
  const auto path = make_file("ordered");
  constexpr unsigned int n_frames = 200;

  {
    // Depth 1: more frames than slots by a wide margin, so the producer is forced to wait
    // repeatedly. Nothing may be dropped.
    HDF5FrameWriter writer(1);
    REQUIRE(writer.asynchronous());
    for (unsigned int i = 0; i < n_frames; ++i)
      writer.submit(counting_frame(path.string(), i));
    writer.finish();
  }

  CHECK(count_frames(path) == n_frames);

  auto file = DiFfRG::hdf5::File::open(path.string(), DiFfRG::hdf5::Access::ReadOnly);
  auto frames = file.root().open_group("frames");
  for (unsigned int i = 0; i < n_frames; ++i) {
    REQUIRE(frames.has_dataset(std::to_string(i)));
    double value = -1.;
    frames.open_dataset(std::to_string(i)).read(value);
    CHECK(value == static_cast<double>(i));
  }

  std::filesystem::remove(path);
}

TEST_CASE("A drained file is complete and readable mid-run", "[output][hdf5][writer]")
{
  const auto path = make_file("midrun");

  HDF5FrameWriter writer(1);
  for (unsigned int i = 0; i < 8; ++i)
    writer.submit(counting_frame(path.string(), i));

  // The crash-safety contract: after a drain the file is closed on disk and a reader can open
  // it while the writer stays alive for more frames.
  writer.drain();
  CHECK(count_frames(path) == 8);

  for (unsigned int i = 8; i < 12; ++i)
    writer.submit(counting_frame(path.string(), i));
  writer.finish();

  CHECK(count_frames(path) == 12);
  std::filesystem::remove(path);
}

TEST_CASE("A failed write surfaces on the submitting thread, naming its frame",
          "[output][hdf5][writer][error]")
{
  const auto path = make_file("failure");

  HDF5FrameWriter writer(1);

  // A frame whose file cannot be opened at all. The failure happens on the worker; it has to
  // come back to whoever is driving the run.
  HDF5Frame doomed = counting_frame("/nonexistent-directory-for-diffrg/never.h5", 7);
  doomed.time = 12.5;
  writer.submit(std::move(doomed));

  std::string message;
  try {
    writer.drain();
    FAIL("drain() should have rethrown the worker's failure");
  } catch (const std::exception &error) {
    message = error.what();
  }
  // Without the time there is no way to tell which frame died: by the time it surfaces the main
  // thread has moved on.
  CHECK(message.find("12.5") != std::string::npos);

  SECTION("the error is reported once, then cleared")
  {
    // A second drain must not replay the same failure forever.
    CHECK_NOTHROW(writer.drain());
  }

  SECTION("finish() after a reported error is clean")
  {
    CHECK_NOTHROW(writer.finish());
  }

  std::filesystem::remove(path);
}

TEST_CASE("Depth zero runs inline and starts no thread", "[output][hdf5][writer][sync]")
{
  const auto path = make_file("inline");

  HDF5FrameWriter writer(0);
  CHECK_FALSE(writer.asynchronous());

  for (unsigned int i = 0; i < 5; ++i) {
    writer.submit(counting_frame(path.string(), i));
    // Synchronous means committed on return -- this is what `asynchronous = false` buys, and
    // why it stays available for runs that cannot afford to lose a queued frame.
    CHECK(count_frames(path) == i + 1);
  }

  CHECK_NOTHROW(writer.finish());
  std::filesystem::remove(path);
}

TEST_CASE("An empty frame is not written at all", "[output][hdf5][writer]")
{
  const auto path = make_file("empty");

  HDF5FrameWriter writer(1);
  HDF5Frame empty;
  empty.path = path.string();
  writer.submit(std::move(empty));
  writer.finish();

  // No open, no close, no zero-length series group.
  CHECK(count_frames(path) == 0);
  std::filesystem::remove(path);
}
