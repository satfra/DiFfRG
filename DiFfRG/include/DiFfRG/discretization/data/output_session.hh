#pragma once

#include <DiFfRG/common/run_reporter.hh>
#include <DiFfRG/discretization/common/eom.hh>
#include <DiFfRG/discretization/data/csv_output.hh>
#include <DiFfRG/discretization/data/diagnostic_port.hh>
#include <DiFfRG/discretization/data/fe_output.hh>
#include <DiFfRG/discretization/data/hdf5_output.hh>
#include <DiFfRG/discretization/data/hdf5_writer.hh>
#include <DiFfRG/discretization/data/output_path.hh>
#include <DiFfRG/discretization/data/output_settings.hh>
#include <DiFfRG/discretization/data/output_timings.hh>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace DiFfRG
{
  using namespace dealii;

  namespace internal
  {
    template <typename...> inline constexpr bool output_frame_api_removed = false;
  }

  template <uint dim, typename VectorType> class OutputSession_impl;

  /**
   * @brief What a model or assembler writes into during one output frame; see OutputSession for the files.
   *
   * Created by OutputSession::write_frame and handed to the callback; it cannot be copied or kept. Everything added
   * to it is written when the callback returns. If the callback or the write throws, the session refuses all further
   * frames.
   *
   * On ranks other than 0 the frame discards everything added to it, but the callback still runs there: see
   * OutputSession_impl::write_frame.
   */
  template <uint dim, typename VectorType> class OutputFrame
  {
  public:
    /// Adds FE fields to a field series; see FEOutput::attach. Discards them on a non-writing rank.
    class FieldCollector
    {
    public:
      /// Add a scalar field named @p name.
      void attach(const DoFHandler<dim> &dof_handler, const VectorType &solution, const std::string &name)
      {
        if (sink) sink->attach(dof_handler, solution, name);
      }
      /// Add a vector-valued field, one name per component.
      void attach(const DoFHandler<dim> &dof_handler, const VectorType &solution, const std::vector<std::string> &names)
      {
        if (sink) sink->attach(dof_handler, solution, names);
      }

    private:
      explicit FieldCollector(FEOutput<dim, VectorType> *sink) : sink(sink) {}
      FEOutput<dim, VectorType> *sink;
      friend class OutputFrame;
    };

    /// Adds one row to a CSV table; see CsvOutput. Discards it on a non-writing rank.
    class TableRecord
    {
    public:
      /// Set column @p name of this frame's row.
      void value(const std::string &name, double value)
      {
        if (sink) sink->value(name, value);
      }
      /// See CsvOutput::set_Lambda; by default the table uses `/physical/Lambda`.
      void set_Lambda(double Lambda)
      {
        if (sink) sink->set_Lambda(Lambda);
      }

    private:
      explicit TableRecord(CsvOutput *sink) : sink(sink) {}
      CsvOutput *sink;
      friend class OutputFrame;
    };

    /// Adds scalars and maps to an HDF5 file; see HDF5Output. Discards them on a non-writing rank.
    class Hdf5Record
    {
    public:
      /// See HDF5Output::scalar.
      template <typename... Args> void scalar(Args &&...args)
      {
        if (sink) sink->scalar(std::forward<Args>(args)...);
      }
      /// See HDF5Output::map.
      template <typename... Args> void map(Args &&...args)
      {
        if (sink) sink->map(std::forward<Args>(args)...);
      }

    private:
      explicit Hdf5Record(HDF5Output *sink) : sink(sink) {}
      HDF5Output *sink;
      friend class OutputFrame;
    };

    OutputFrame(const OutputFrame &) = delete;
    OutputFrame &operator=(const OutputFrame &) = delete;
    OutputFrame(OutputFrame &&) = delete;
    OutputFrame &operator=(OutputFrame &&) = delete;

    /// The main field series (VTK `<run>.pvd`, HDF5 group `/FE`).
    FieldCollector fields() { return FieldCollector(writes() ? &session.fe_out : nullptr); }
    /**
     * @brief An additional field series @p name, created on first use: VTK `<run>_<name>.pvd` (if `/output/vtk`),
     * HDF5 group `/<name>` in the run's `.h5` file (if `/output/hdf5`).
     *
     * @p name is a single path component and not one of FE, potential, eom_potential, scalars, maps, coordinates.
     */
    FieldCollector fields(const std::string &name)
    {
      return FieldCollector(writes() ? &session.field_output(name) : nullptr);
    }
    /**
     * @brief The CSV table `<run>_<name>`, e.g. table("data.csv") writes `<run>_data.csv`. Created on first use;
     * from then on every frame must write the same columns.
     */
    TableRecord table(const std::string &name) { return TableRecord(writes() ? &session.csv(name) : nullptr); }
    /// The extra HDF5 file `<run>_<name>.h5`, created on first use; written regardless of `/output/hdf5`.
    Hdf5Record hdf5(const std::string &name) { return Hdf5Record(writes() ? &session.hdf5(name) : nullptr); }
    /// The run's HDF5 file `<run>.h5`; throws std::logic_error if `/output/hdf5` is false.
    Hdf5Record hdf5()
    {
      // Checked on every rank, so that a misconfigured run fails everywhere rather than on rank 0 alone.
      if (!session.use_hdf5) throw std::logic_error("OutputSession::hdf5: HDF5 output is disabled.");
      return Hdf5Record(writes() ? &session.hdf5() : nullptr);
    }
    /// Handle for diagnostic CSV tables `<run>_<table>`, usable outside of frames.
    DiagnosticPort diagnostics() const { return session.diagnostic_port(); }

    template <typename Name>
    [[deprecated("OutputFrame::csv(name) was removed. Use auto table = output.table(name) instead")]] CsvOutput &
    csv(Name &&name)
    {
      static_assert(internal::output_frame_api_removed<Name>,
                    "OutputFrame::csv(name) was removed. Use auto table = output.table(name) instead");
      return session.csv(std::forward<Name>(name));
    }

    template <typename Lambda>
    [[deprecated("OutputFrame::set_Lambda(value) was removed. Call set_Lambda(value) on the TableRecord returned by "
                 "output.table(name) instead")]] void
    set_Lambda(Lambda &&)
    {
      static_assert(internal::output_frame_api_removed<Lambda>,
                    "OutputFrame::set_Lambda(value) was removed. Call set_Lambda(value) on the TableRecord returned "
                    "by output.table(name) instead");
    }

    template <typename LegacyVectorType = VectorType>
    [[deprecated("OutputFrame::fe_output() was removed. Use auto fields = output.fields(); fields.attach(...) "
                 "instead")]] FEOutput<dim, VectorType> &
    fe_output()
    {
      static_assert(internal::output_frame_api_removed<LegacyVectorType>,
                    "OutputFrame::fe_output() was removed. Use auto fields = output.fields(); fields.attach(...) "
                    "instead");
      return session.fe_out;
    }

    /// Write the reconstructed EoM potential of @p result (if any) to the `eom_potential` series.
    void attach_eom_potential(EoMResult<dim, typename VectorType::value_type> result)
    {
      session.attach_eom_potential(std::move(result));
    }

    /// Write the raw potential to the `potential` series; at most once per frame.
    void attach_raw_potential(ReconstructedRawPotential<dim, typename VectorType::value_type> potential)
    {
      session.attach_raw_potential(std::move(potential));
    }

    /// Whether a raw potential attached to this frame would be written anywhere (VTK or HDF5).
    bool writes_raw_potential() const
    {
      if constexpr (dim > 0)
        return !session.potential_fe_out.will_discard();
      else
        return false;
    }

    /**
     * @brief Write @p values (one inner vector per row) as CSV to `<root>/<name>`, immediately.
     *
     * Unlike table(), the file name is not prefixed with the run name. @p header is written only if @p append is
     * false.
     */
    void dump_table(const std::string &name, const std::vector<std::vector<double>> &values, bool append = false,
                    const std::vector<std::string> &header = {})
    {
      if (writes()) session.dump_to_csv(name, values, append, header);
    }

    /// The run name (`/output/name`).
    const std::string &run_name() const noexcept { return session.output_name; }

    /// Declare a readout id for this frame; throws if it is empty or was already declared.
    void register_readout(const std::string &id)
    {
      if (id.empty()) throw std::invalid_argument("OutputFrame::register_readout: id must not be empty.");
      if (!readout_ids.insert(id).second)
        throw std::invalid_argument("OutputFrame::register_readout: duplicate id '" + id + "'.");
    }

  private:
    explicit OutputFrame(OutputSession_impl<dim, VectorType> &session) : session(session) {}
    /// Whether this rank writes what is added to the frame.
    bool writes() const noexcept { return session.active; }
    OutputSession_impl<dim, VectorType> &session;
    std::set<std::string> readout_ids;
    friend class OutputSession_impl<dim, VectorType>;
  };

  // clang-format off
  /**
   * @brief All output of one run. Applications use the alias OutputSession<Assembler>.
   *
   * Construct one and pass it to the timestepper, which calls write_frame at every output time; the assembler and
   * model fill the frame (AbstractAssembler::attach_data_output, Model::readouts). Only MPI rank 0 writes. VTK and HDF5
   * frames are written on background threads.
   *
   * ```cpp
   * DiFfRG::OutputSession<Assembler> output(config); // files go to /output/folder
   * ```
   * A default-constructed session writes to a new temporary folder that is deleted at the end.
   *
   * ### Files
   * With R = `/output/folder` (default "./"), N = `/output/name` (default "output") and F = `/output/field_directory`
   * (default N):
   *
   * | File                                   | Content | Written if |
   * |----------------------------------------|---------|------------|
   * | `R/N.pvd`, `R/F/N_NNNNNN.vtu`          | FE fields, one .vtu per frame NNNNNN | `/output/vtk` |
   * | `R/N_potential.pvd` (+ .vtu)           | raw potential (attach_raw_potential) | `/output/vtk` |
   * | `R/N_eom_potential.pvd` (+ .vtu)       | EoM potential (attach_eom_potential) | `/output/vtk` |
   * | `R/N_X.pvd` (+ .vtu)                   | field series fields("X") | `/output/vtk` |
   * | `R/N.h5`                               | `/config`, `/scalars`, `/maps`, `/coordinates`, the field series `/FE`, `/potential`, `/eom_potential`, `/X`; see HDF5Output | `/output/hdf5` |
   * | `R/N_X.h5`                             | hdf5("X"), same layout without field series | always |
   * | `R/N_X` (e.g. `R/N_data.csv`)          | table("X"), see CsvOutput | always |
   * | `R/N.log`                              | the run log, filtered by `/output/log_level` | always |
   * | `R/N.log.json`                         | the parameters | `/output/json`, or `/output/hdf5` false |
   * | `R/N_snapshot_nnn.h5`                  | flow snapshots, see SnapshotData | snapshots requested |
   * | `R/N_output_timings.csv`               | cost of each output frame | `/output/verbosity` ≥ 3 |
   *
   * The output keys are listed in Config::OutputSettings.
   */
  // clang-format on
  template <uint dim, typename VectorType> class OutputSession_impl
  {
  public:
    /// Output into a temporary folder that is deleted at the end, with default settings.
    OutputSession_impl() : OutputSession_impl(OutputPath::temporary(), Config::OutputSettings{}) {}
    /// Paths (OutputPath) and settings (Config::OutputSettings) from the parameters.
    explicit OutputSession_impl(const ConfigTree &config)
        : OutputSession_impl(OutputPath(config), Config::OutputSettings(config))
    {
    }
    /// Explicit paths and settings.
    explicit OutputSession_impl(OutputPath path, Config::OutputSettings settings = {});
    /// Explicit paths, settings from the parameters.
    OutputSession_impl(OutputPath path, const ConfigTree &config)
        : OutputSession_impl(std::move(path), Config::OutputSettings(config))
    {
    }
    /// finish(); the HDF5 files get `crashed` = 1 if an exception is propagating.
    ~OutputSession_impl() noexcept;

    OutputSession_impl(const OutputSession_impl &) = delete;
    OutputSession_impl &operator=(const OutputSession_impl &) = delete;
    OutputSession_impl(OutputSession_impl &&) = delete;
    OutputSession_impl &operator=(OutputSession_impl &&) = delete;

    /**
     * @brief Write one frame at RG time @p time: call `contributor(OutputFrame &)`, then write what it added.
     *
     * A second frame at the same time as the previous one is skipped. After an error, every further call rethrows it.
     *
     * Collective: call it on every rank. Ranks other than 0 write nothing, but still run @p contributor (on a frame
     * that discards what is added to it), because the readouts in it may call collective code such as map(), or
     * update model state that has to stay the same on every rank.
     */
    template <typename Contributor> void write_frame(const double time, Contributor &&contributor)
    {
      std::scoped_lock lock(submission_mutex);
      rethrow_deferred_error();
      if (terminal_error) std::rethrow_exception(terminal_error);
      if (finished) throw std::logic_error("OutputSession::write_frame: session has already been finished.");
      // Consecutive run segments (snapshots) both write the frame at their common time; keep the first.
      if (last_frame_time && std::abs(time - *last_frame_time) <= 1e-12 * std::max(1., std::abs(time))) return;

      if (!active) {
        try {
          OutputFrame<dim, VectorType> frame(*this);
          std::forward<Contributor>(contributor)(frame);
          last_frame_time = time;
        } catch (...) {
          terminal_error = std::current_exception();
          throw;
        }
        return;
      }

      try {
        current_frame = internal::FrameTimings{};
        internal::ScopedTimer total_timer(current_frame.total);
        {
          const auto solves_before = internal::potential_solve_stats();
          internal::ScopedTimer contributor_timer(current_frame.contributor);
          OutputFrame<dim, VectorType> frame(*this);
          std::forward<Contributor>(contributor)(frame);
          contributor_timer.stop();
          const auto &solves_after = internal::potential_solve_stats();
          current_frame.potential_solve_count += solves_after.solves - solves_before.solves;
          current_frame.potential_solves += solves_after.seconds - solves_before.seconds;
        }
        flush_frame(time);
        total_timer.stop();
        record_frame(time);
        last_frame_time = time;
      } catch (...) {
        terminal_error = std::current_exception();
        throw;
      }
    }

    /// See OutputFrame::diagnostics.
    DiagnosticPort diagnostic_port() const { return diagnostics_port; }
    /// The run's logger.
    ReportPort report_port() const { return run_reporter.port(); }

    /// Wait for all pending writes; rethrow the first error. The session stays open.
    void drain();

    /// Wait for all pending writes, mark the HDF5 files finished, and close the session.
    void finish() { finish_impl(/* crashed = */ false); }
    /// Set \f$\Lambda\f$ of the CSV tables' `k [GeV]` column (default: `/physical/Lambda`); before the first frame.
    void set_Lambda(double Lambda);
    /// The run name.
    const std::string &run_name() const noexcept { return output_name; }
    /// The output paths.
    const OutputPath &path() const noexcept { return output_path; }
    /// Whether this rank writes (rank 0).
    bool is_writer_rank() const noexcept { return active; }

  private:

    FEOutput<dim, VectorType> &fe_output() { return fe_out; }
    FEOutput<dim, VectorType> &field_output(const std::string &name);
    template <typename FieldVectorType>
    void configure_field_output(FEOutput<dim, FieldVectorType> &sink, const std::string &group_name);
    void attach_raw_potential(ReconstructedRawPotential<dim, typename VectorType::value_type> potential);
    void attach_eom_potential(EoMResult<dim, typename VectorType::value_type> result);
    CsvOutput &csv(const std::string &name);
    HDF5Output &hdf5(const std::string &name);
    HDF5Output &hdf5();
    void flush_frame(double time);
    /// Add `current_frame` to the run totals and, at verbosity >= 3, report it.
    void record_frame(double time);
    void dump_to_csv(const std::string &name, const std::vector<std::vector<double>> &values, bool append,
                     const std::vector<std::string> &header);
    void rethrow_deferred_error();

    OutputPath output_path;
    Config::OutputSettings settings;
    /// finish() and the destructor; @p crashed is recorded in the HDF5 files.
    void finish_impl(bool crashed);

    const std::string top_folder;
    const std::string output_name;
    const std::string output_folder;
    double Lambda = -1.;
    bool active = true;
    bool finished = false;
    std::exception_ptr deferred_error;
    std::exception_ptr terminal_error;
    mutable std::mutex submission_mutex;

    RunReporter run_reporter;
    /// Declared before the sinks that use it, so it is destroyed after them.
    internal::HDF5FrameWriter hdf5_writer;
    FEOutput<dim, VectorType> fe_out;
    std::optional<ReconstructedRawPotential<dim, typename VectorType::value_type>> pending_raw_potential;
    FEOutput<dim, dealii::Vector<typename VectorType::value_type>> potential_fe_out;
    std::vector<ReconstructedEoMPotential<dim, typename VectorType::value_type>> pending_eom_potentials;
    FEOutput<dim, dealii::Vector<typename VectorType::value_type>> eom_potential_fe_out;
    std::map<std::string, FEOutput<dim, VectorType>> named_fe_outs;
    std::set<std::string> pending_named_fe_outs;
    std::map<std::string, CsvOutput> csv_files;

    bool use_hdf5;
    const std::string filename_h5;
    std::map<std::string, HDF5Output> h5_files;
    /// Time of the last frame written, to drop an immediate duplicate; see write_frame().
    std::optional<double> last_frame_time;
    DiagnosticPort diagnostics_port;

    /// Timings of the frame being written.
    internal::FrameTimings current_frame;
    internal::OutputTimings timings;

    friend class OutputFrame<dim, VectorType>;
  };

  /**
   * @brief The output session of a run, see OutputSession_impl.
   *
   * @tparam AssemblerOrDiscretization Any type with `dim` and `VectorType`, usually the Assembler.
   */
  template <typename AssemblerOrDiscretization>
  using OutputSession =
      OutputSession_impl<AssemblerOrDiscretization::dim, typename AssemblerOrDiscretization::VectorType>;
} // namespace DiFfRG
