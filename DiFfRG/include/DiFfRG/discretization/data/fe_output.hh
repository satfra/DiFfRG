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
   * @brief One series of FE fields, written once per frame as VTK (a `.vtu` file per frame, collected in a `.pvd` time
   * series) and/or into an HDF5 file.
   *
   * Applications use it through OutputFrame::fields. Per frame: attach() the fields, then flush(time). Files:
   * - VTK: `<top_folder>/<output_name>.pvd` and `<top_folder>/<output_folder>/<output_name>_NNNNNN.vtu`, NNNNNN the
   *   frame number; written on background threads, with at most `/output/max_pending_frames` (at least 2) frames in
   *   flight.
   * - HDF5 (if set_hdf5_output was called): group `<group_name>/NNNNNN` with the dataset `nodes` (the output points,
   *   shape n_nodes × dim) and one dataset per field (its values at the nodes).
   *
   * Each cell is subdivided `/discretization/output_subdivisions` times for the output.
   *
   * @tparam dim Spatial dimension (for dim = 0 nothing is written).
   * @tparam VectorType Vector type of the attached fields.
   */
  template <uint dim, typename VectorType> class FEOutput
  {
    /// The vector type handed to deal.II's DataOut: a serial copy for distributed vectors.
    using OutputVectorType =
        std::conditional_t<is_distributed_la<VectorType>, dealii::Vector<get_type::NumberType<VectorType>>, VectorType>;

  public:
    /**
     * @brief Set up the series; creates the folders.
     *
     * @param top_folder Folder of the .pvd file.
     * @param output_name Name of the series: the .pvd file and the prefix of the .vtu files.
     * @param output_folder Folder of the .vtu files, relative to @p top_folder.
     * @param settings Output settings: write_vtk, max_pending_frames, subdivisions, asynchronous.
     * @param active False on ranks that do not write (everything attached is discarded).
     */
    FEOutput(std::string top_folder, std::string output_name, std::string output_folder,
             const Config::OutputSettings &settings, bool active = true);
    /// Settings taken from @p config.
    [[deprecated(
        "Use FEOutput(top_folder, output_name, output_folder, Config::OutputSettings(config), active) instead")]]
    FEOutput(std::string top_folder, std::string output_name, std::string output_folder, const ConfigTree &config,
             bool active = true)
        : FEOutput(std::move(top_folder), std::move(output_name), std::move(output_folder),
                   Config::OutputSettings(config), active)
    {
    }

    /// finish(), ignoring errors.
    ~FEOutput() noexcept;

    /// Wait for the pending frames to be written; rethrow the first writer error.
    void drain();

    /// Wait for the pending frames and close the series.
    void finish();

    /**
     * @brief Also write the frames into @p output (not owned), under the group @p group_name.
     *
     * @param output The HDF5 file.
     * @param session_closes_file true if the owner flushes @p output after each frame (as OutputSession does);
     * false to let flush() write the HDF5 frame itself.
     * @param group_name Top-level group of this series in the file.
     */
    void set_hdf5_output(HDF5Output *output, bool session_closes_file = false, std::string group_name = "FE");

    /// True if nothing is written: VTK off (or inactive rank) and no HDF5 file set.
    bool will_discard() const { return !save_vtk && hdf5_output == nullptr; }

    /// Number of frames still being written in the background.
    std::size_t pending_workers() const { return output_threads.size(); }

    /**
     * @brief Add a scalar field to the current frame (copied).
     *
     * @param dof_handler The dofs of @p solution.
     * @param solution The field; for a distributed vector, every global dof must be readable (a SolutionView).
     * @param name Name of the field in the output.
     */
    void attach(const DoFHandler<dim> &dof_handler, const VectorType &solution, const std::string &name);

    /**
     * @brief Add a vector-valued field to the current frame (copied), one output field per component.
     *
     * @param dof_handler The dofs of @p solution.
     * @param solution The field; for a distributed vector, every global dof must be readable (a SolutionView).
     * @param names One name per component.
     */
    void attach(const DoFHandler<dim> &dof_handler, const VectorType &solution, const std::vector<std::string> &names);

    /**
     * @brief Write the attached fields as the next frame, at RG time @p time, and start a new frame.
     */
    void flush(double time);

    /// The timings accumulated since the last call.
    internal::FrameTimings take_frame_timings()
    {
      const internal::FrameTimings taken = frame_timings;
      frame_timings = internal::FrameTimings{};
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
    // Frame of each worker thread, so the main thread can clear its DataOut after joining (clearing frees Kokkos
    // memory, which must happen on the main thread).
    std::list<uint> output_thread_series;
    // Copies of the attached fields per pending frame; owned (not pooled) so they are freed on the main thread.
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

    /// DoFHandler on a serial copy of a parallel mesh, and the map from the original dof numbering to its numbering.
    struct SerialMirrorDoFs {
      std::unique_ptr<DoFHandler<safe_dim>> dof_handler;
      std::vector<types::global_dof_index> to_mirror;
      boost::signals2::scoped_connection connection;
      bool stale = true;
    };
    SerialMirrorDoFs mirror_dofs;

    /// The DoFHandler to hand DataOut, and the values permuted into its numbering.
    const DoFHandler<safe_dim> &prepare_output_vector(const DoFHandler<dim> &dof_handler, const VectorType &solution,
                                                      OutputVectorType &target);

    void update_buffers();
    void retire_oldest();
    void reserve_bytes(std::size_t bytes);

    HDF5Output *hdf5_output = nullptr;
    bool hdf5_closed_by_session = false;
    std::string hdf5_group_name = "FE";

    bool save_vtk;

    internal::FrameTimings frame_timings;
  };

  /// Zero-dimensional field space: there are no fields, nothing is written.
  template <typename VectorType> class FEOutput<0, VectorType>
  {
  public:
    /**
     * @brief Set up the series; creates the folders.
     *
     * @param top_folder Folder of the .pvd file.
     * @param output_name Name of the series: the .pvd file and the prefix of the .vtu files.
     * @param output_folder Folder of the .vtu files, relative to @p top_folder.
     * @param settings Output settings: write_vtk, max_pending_frames, subdivisions, asynchronous.
     * @param active False on ranks that do not write (everything attached is discarded).
     */
    FEOutput(std::string top_folder, std::string output_name, std::string output_folder,
             const Config::OutputSettings &settings, bool active = true);
    /// Settings taken from @p config.
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
    internal::FrameTimings take_frame_timings() { return {}; }
  };

} // namespace DiFfRG
