#pragma once

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/la_policy.hh>
#include <DiFfRG/discretization/common/serial_mirror.hh>
#include <DiFfRG/discretization/data/hdf5_output.hh>
#include <DiFfRG/discretization/data/output_settings.hh>
#include <DiFfRG/discretization/data/output_timings.hh>

// external libraries
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/numerics/data_out.h>

#include <hdf5lib/hdf5.hh>

// standard library
#include <condition_variable>
#include <exception>
#include <list>
#include <mutex>
#include <thread>

namespace DiFfRG
{
  using namespace dealii;

  /**
   * @brief A class to output finite element data to disk as .vtu files and .pvd time series.
   *
   * @tparam dim Dimension of the problem
   * @tparam VectorType Type of the vector
   */
  template <uint dim, typename VectorType> class FEOutput
  {
    /**
     * @brief What the DataOut actually gets handed.
     *
     * Never a distributed vector. DataOut::add_data_vector() routes every vector type through
     * internal::DataOutImplementation::CreateVectors::create_dof_vector(), which builds a ghosted
     * LinearAlgebra::distributed::BlockVector on dof_handler.get_mpi_communicator() and calls
     * update_ghost_values() -- two collectives, in a code path OutputSession runs on rank 0 alone.
     * Under the replicated-mesh policy rank 0 already holds every dof, so it copies the replica into
     * a plain serial vector and attaches that to a serial mirror DoFHandler instead. Nothing
     * communicates and the whole domain still gets written.
     */
    using OutputVectorType =
        std::conditional_t<is_distributed_la<VectorType>, dealii::Vector<get_type::NumberType<VectorType>>, VectorType>;

  public:
    /**
     * @brief Construct a new FEOutput object
     *
     * @param dof_handler DoFHandler associated with the solution
     * @param top_folder Folder where the output will be written, i.e. the folder containing the .pvd file.
     * @param output_name Name of the output, i.e. the name of the .pvd file.
     * @param output_folder Folder where the .vtu files will be saved. Should be relative to top_folder.
     * @param subdivisions Number of subdivisions of the cells in the .vtu files.
     */
    FEOutput(std::string top_folder, std::string output_name, std::string output_folder,
             const Config::OutputSettings &settings, bool active = true);
    [[deprecated(
        "Use FEOutput(top_folder, output_name, output_folder, Config::OutputSettings(config), active) instead")]]
    FEOutput(std::string top_folder, std::string output_name, std::string output_folder, const ConfigTree &config,
             bool active = true)
        : FEOutput(std::move(top_folder), std::move(output_name), std::move(output_folder),
                   Config::OutputSettings(config), active)
    {
    }

    FEOutput();
    ~FEOutput() noexcept;

    /** Join pending writers and surface the first worker error without closing the output. */
    void drain();

    /** Drain pending writers, release terminal resources, and close the output. */
    void finish();

    /**
     * @brief Route the finite-element frame data into an HDF5 file owned by someone else.
     *
     * @param session_closes_file When true, flush() leaves the file open and relies on the
     * owner to close it once per frame -- which is what OutputSession does, since it flushes
     * its HDF5Output sinks after this one. Standalone users, who may never call
     * HDF5Output::flush() at all, keep the default and get a close per frame.
     * @param group_name Top-level HDF5 group that owns this field series.
     */
    void set_hdf5_output(HDF5Output *output, bool session_closes_file = false, std::string group_name = "FE");

    /** True when flush() would throw everything attached to it away. */
    bool will_discard() const { return !save_vtk && hdf5_output == nullptr; }

    /** Number of writer threads that have not been joined yet. */
    std::size_t pending_workers() const { return output_threads.size(); }

    /**
     * @brief Attach a solution to the output.
     *
     * @param dof_handler The DoFHandler associated with the solution.
     * @param solution The solution to attach.
     * @param name The name of the solution.
     */
    void attach(const DoFHandler<dim> &dof_handler, const VectorType &solution, const std::string &name);

    /**
     * @brief Attach a solution to the output.
     *
     * @param dof_handler The DoFHandler associated with the solution.
     * @param solution The solution to attach.
     * @param names The names of each component of the solution.
     */
    void attach(const DoFHandler<dim> &dof_handler, const VectorType &solution, const std::vector<std::string> &names);

    /**
     * @brief Flush all attached solutions to disk.
     *
     * @param time The time tag to attach to the solution.
     */
    void flush(double time);

    /** Timing buckets accumulated since the last call, and reset by it. */
    FrameTimings take_frame_timings()
    {
      const FrameTimings taken = frame_timings;
      frame_timings = FrameTimings{};
      return taken;
    }

  private:
    const std::string top_folder;
    const std::string output_name;
    const std::string output_folder;
    const std::string filename_pvd;
    const uint buffer_size;
    const std::size_t max_pending_bytes;

    uint series_number, subdivisions;
    std::vector<std::pair<double, std::string>> time_series;

    constexpr static uint safe_dim = dim == 0 ? 1 : dim;
    std::vector<DataOut<safe_dim>> data_outs;
    std::list<std::thread> output_threads;
    // series number each worker thread is responsible for, kept in lockstep with output_threads so
    // the main thread can clear() the corresponding DataOut once the worker has been joined. Clearing
    // a DataOut frees its Kokkos-backed vectors, which (with deal.II's Kokkos host backend) must not
    // happen on a worker thread concurrently with the main thread's own Kokkos work.
    std::list<uint> output_thread_series;
    // Owned outright rather than drawn from a GrowingVectorMemory pool. A pooled vector is not
    // destroyed when the frame retires, it is handed back to a process-wide pool that lives until
    // InitFinalize::finalize() tears it down -- i.e. these buffers would be freed during MPI/Kokkos
    // shutdown, concurrently with whatever else is being finalised. That showed up as an
    // intermittent segfault in GrowingVectorMemory::release_unused_memory() after the run had
    // already printed its final line. Owning them means they die in retire_oldest()/drain(), on the
    // main thread, while the run is still up.
    std::list<std::list<std::unique_ptr<OutputVectorType>>> attached_solutions;
    std::list<std::size_t> attached_bytes;
    std::size_t pending_bytes = 0;

    // serializes worker-thread access to time_series and the shared .pvd file
    std::mutex output_mutex;
    std::condition_variable output_condition;
    uint next_published_series = 0;
    bool publication_failed = false;
    // captures any exception thrown inside a worker thread so it can be re-raised on the
    // main thread at the next flush() (or at destruction, if no other throw is unwinding)
    std::mutex exception_mutex;
    std::exception_ptr stored_exception;
    bool finished = false;

    /**
     * @brief Serial mirror of the solution DoFHandler, plus the map from its dof numbering to ours.
     *
     * The numbering genuinely differs: a DoFHandler on a parallel::shared::Triangulation renumbers
     * so that each rank owns a contiguous range, so global index i means different nodes in the two
     * handlers as soon as there is more than one rank. (At one rank they coincide, which is exactly
     * why single-rank distributed runs looked fine.) The permutation is built cell by cell -- the
     * two triangulations are structurally identical, so cells match by level and index.
     */
    struct SerialMirrorDoFs {
      std::unique_ptr<DoFHandler<safe_dim>> dof_handler;
      std::vector<types::global_dof_index> to_mirror;
      boost::signals2::scoped_connection connection;
      bool stale = true;
    };
    SerialMirrorDoFs mirror_dofs;

    /** The DoFHandler to hand DataOut, and the values permuted into its numbering. */
    const DoFHandler<safe_dim> &prepare_output_vector(const DoFHandler<dim> &dof_handler, const VectorType &solution,
                                                      OutputVectorType &target);

    void update_buffers();
    void retire_oldest();
    void reserve_bytes(std::size_t bytes);

    HDF5Output *hdf5_output = nullptr;
    bool hdf5_closed_by_session = false;
    std::string hdf5_group_name = "FE";

    bool save_vtk;

    FrameTimings frame_timings;
  };

  template <typename VectorType> class FEOutput<0, VectorType>
  {
  public:
    /**
     * @brief Construct a new FEOutput object
     *
     * @param dof_handler DoFHandler associated with the solution
     * @param top_folder Folder where the output will be written, i.e. the folder containing the .pvd file.
     * @param output_name Name of the output, i.e. the name of the .pvd file.
     * @param output_folder Folder where the .vtu files will be saved. Should be relative to top_folder.
     * @param subdivisions Number of subdivisions of the cells in the .vtu files.
     */
    FEOutput(std::string top_folder, std::string output_name, std::string output_folder,
             const Config::OutputSettings &settings, bool active = true);
    [[deprecated(
        "Use FEOutput(top_folder, output_name, output_folder, Config::OutputSettings(config), active) instead")]]
    FEOutput(std::string top_folder, std::string output_name, std::string output_folder, const ConfigTree &config,
             bool active = true)
        : FEOutput(std::move(top_folder), std::move(output_name), std::move(output_folder),
                   Config::OutputSettings(config), active)
    {
    }
    void drain() {}
    void finish() {}
    FrameTimings take_frame_timings() { return {}; }
  };

} // namespace DiFfRG
