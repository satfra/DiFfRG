#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/discretization/data/output_session.hh>

#include <deal.II/lac/vector.h>

#include <filesystem>

// Without MPI, deal.II's MPI_COMM_WORLD stand-in lives in namespace dealii; DiFfRG/common/mpi.hh re-exports it.
using namespace DiFfRG;

// Readouts in a frame's contributor may call collective code (map()) or update model state, so the contributor has
// to run on every rank, while only rank 0 writes.
TEST_CASE("OutputSession runs the frame contributor on every rank and writes on rank 0 only", "[output][session][mpi]")
{
  DiFfRG::Init();
  const bool writer = DiFfRG::MPI::rank(MPI_COMM_WORLD) == 0;

  // One folder per rank, so that what a rank creates can be told apart.
  auto path = DiFfRG::OutputPath::temporary();
  DiFfRG::Config::OutputSettings settings;
  settings.Lambda = 1.;
  settings.write_vtk = false;
  settings.write_hdf5 = true;

  int calls = 0;
  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, settings);
    CHECK(output.is_writer_rank() == writer);

    const auto contribute = [&](auto &frame) {
      ++calls;
      frame.table("data.csv").value("a", 1.);
      frame.hdf5().scalar("a", 1.);
      frame.hdf5("extra").scalar("b", 2.);
      frame.dump_table("dump.csv", {{1., 2.}});
    };
    output.write_frame(0., contribute);
    output.write_frame(0., contribute); // the same time again: skipped, on every rank alike
    output.write_frame(1., contribute);
  }

  CHECK(calls == 2);
  CHECK_FALSE(DiFfRG::MPI::any_of(MPI_COMM_WORLD, calls != 2));

  const auto &root = path.root();
  const auto run = path.run_name();
  CHECK(std::filesystem::exists(root / (run + "_data.csv")) == writer);
  CHECK(std::filesystem::exists(root / (run + ".h5")) == writer);
  CHECK(std::filesystem::exists(root / (run + "_extra.h5")) == writer);
  CHECK(std::filesystem::exists(root / "dump.csv") == writer);
}

TEST_CASE("OutputFrame::hdf5() fails on every rank when HDF5 output is off", "[output][session][mpi]")
{
  DiFfRG::Init();
  DiFfRG::Config::OutputSettings settings;
  settings.Lambda = 1.;
  settings.write_vtk = false;
  settings.write_hdf5 = false;

  DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(DiFfRG::OutputPath::temporary(), settings);
  CHECK_THROWS_AS(output.write_frame(0., [](auto &frame) { frame.hdf5().scalar("a", 1.); }), std::logic_error);
}
