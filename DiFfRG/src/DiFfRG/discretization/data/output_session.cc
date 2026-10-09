#include <DiFfRG/discretization/data/output_session.hh>

#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/utils.hh>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <cmath>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace DiFfRG
{
  template <uint dim, typename VectorType>
  template <typename FieldVectorType>
  void OutputSession_impl<dim, VectorType>::configure_field_output(FEOutput<dim, FieldVectorType> &sink,
                                                                   const std::string &group_name)
  {
    if constexpr (dim > 0) {
      if (!active || !use_hdf5) return;

      // A previous frame may still be using the non-thread-safe HDF5 library. Registration is
      // rare and happens on the submitting thread, so wait once before extending the file layout.
      hdf5_writer.drain();
      auto &h5 = h5_files.at(filename_h5);
      {
        auto root_group = h5.get_file().root();
        root_group.create_group(group_name);
      }
      h5.close_file();
      sink.set_hdf5_output(&h5, /*session_closes_file = */ true, group_name);
    }
  }

  template <uint dim, typename VectorType>
  OutputSession_impl<dim, VectorType>::OutputSession_impl(OutputPath path, Config::OutputSettings settings)
      : output_path(std::move(path)), settings(std::move(settings)),
        top_folder(make_folder(output_path.root().string())), output_name(output_path.run_name()),
        output_folder(make_folder(output_path.field_directory().generic_string())),
        active(MPI::rank(MPI_COMM_WORLD) == 0), run_reporter(output_path, this->settings, active),
        hdf5_writer(this->settings.asynchronous && active ? this->settings.hdf5_queue_depth : 0u),
        fe_out(this->top_folder, this->output_name, this->output_folder, this->settings, active),
        potential_fe_out(this->top_folder, this->output_name + "_potential", this->output_folder, this->settings,
                         active),
        eom_potential_fe_out(this->top_folder, this->output_name + "_eom_potential", this->output_folder,
                             this->settings, active),
        use_hdf5(this->settings.write_hdf5), filename_h5(this->output_name + ".h5")
  {
    set_Lambda(this->settings.Lambda);
    diagnostics_port = DiagnosticPort::create(output_path, Lambda, active);
    // With HDF5 on, the configuration already travels with the data in the file's /config group,
    // so the separate copy is only written when asked for or when nothing else records it.
    if (active && (this->settings.write_json || !use_hdf5))
      diagnostics_port.write_text(this->output_name + ".log.json", this->settings.configuration_log);

    if (active && use_hdf5) {
      h5_files.emplace(filename_h5, HDF5Output(this->top_folder, filename_h5, this->settings.configuration_json));
      auto &h5 = h5_files.at(filename_h5);
      h5.set_writer(&hdf5_writer);
      if constexpr (dim > 0) {
        configure_field_output(fe_out, "FE");
        configure_field_output(potential_fe_out, "potential");
        configure_field_output(eom_potential_fe_out, "eom_potential");
      }
    }
  }

  template <uint dim, typename VectorType> OutputSession_impl<dim, VectorType>::~OutputSession_impl() noexcept
  {
    // A destructor runs both at the end of a successful scope and while an exception unwinds one.
    // Either way the session closes in an orderly fashion, so the files are marked finished; the
    // unwinding case is what `crashed` records -- a timestepper that gave up still flushed and
    // closed its output, it just stopped early.
    try {
      finish_impl(/* crashed = */ std::uncaught_exceptions() > 0);
    } catch (...) {
      deferred_error = std::current_exception();
    }
  }

  template <uint dim, typename VectorType>
  void OutputSession_impl<dim, VectorType>::attach_raw_potential(
      ReconstructedRawPotential<dim, typename VectorType::value_type> potential)
  {
    if constexpr (dim > 0) {
      if (pending_raw_potential)
        throw std::logic_error("OutputSession::attach_raw_potential: a raw potential is already attached.");
      if (potential_fe_out.will_discard()) return;
      potential_fe_out.attach(*potential.dof_handler, potential.values, "potential");
      pending_raw_potential.emplace(std::move(potential));
    }
  }

  template <uint dim, typename VectorType>
  void OutputSession_impl<dim, VectorType>::attach_eom_potential(EoMResult<dim, typename VectorType::value_type> result)
  {
    if constexpr (dim > 0) {
      if (!result.potential.has_value()) return;
      if (eom_potential_fe_out.will_discard()) return;
      auto &potential = result.potential.value();
      const std::string name = pending_eom_potentials.empty()
                                   ? "eom_potential"
                                   : "eom_potential_" + std::to_string(pending_eom_potentials.size());
      eom_potential_fe_out.attach(*potential.dof_handler, potential.values, name);
      pending_eom_potentials.push_back(std::move(potential));
    }
  }

  template <uint dim, typename VectorType> CsvOutput &OutputSession_impl<dim, VectorType>::csv(const std::string &name)
  {
    const auto found = csv_files.find(name);
    if (found != csv_files.end()) return found->second;

    const std::string filename = output_name + "_" + OutputPath::checked_relative(name, "table name").generic_string();
    auto [it, inserted] = csv_files.emplace(name, CsvOutput(top_folder, filename));
    if (inserted) it->second.set_Lambda(Lambda);
    return it->second;
  }

  template <uint dim, typename VectorType>
  FEOutput<dim, VectorType> &OutputSession_impl<dim, VectorType>::field_output(const std::string &name)
  {
    const auto checked = OutputPath::checked_relative(name, "field-series name");
    if (checked.has_parent_path())
      throw std::invalid_argument("OutputSession::field_output: field-series name must be one path component.");
    const std::string series_name = checked.generic_string();
    if (series_name == "FE" || series_name == "potential" || series_name == "eom_potential" ||
        series_name == "scalars" || series_name == "maps" || series_name == "coordinates")
      throw std::invalid_argument("OutputSession::field_output: reserved field-series name '" + series_name + "'.");

    auto found = named_fe_outs.find(series_name);
    if (found == named_fe_outs.end()) {
      found =
          named_fe_outs
              .try_emplace(series_name, top_folder, output_name + "_" + series_name, output_folder, settings, active)
              .first;
      configure_field_output(found->second, series_name);
    }
    pending_named_fe_outs.insert(series_name);
    return found->second;
  }

  template <uint dim, typename VectorType> HDF5Output &OutputSession_impl<dim, VectorType>::hdf5(const std::string &name)
  {
    const auto found = h5_files.find(name);
    if (found != h5_files.end()) return found->second;

    const std::string filename = output_name + "_" + OutputPath::checked_relative(name, "HDF5 name").generic_string();
    // Constructing an HDF5Output creates the file and its top-level groups, which are HDF5
    // calls on this thread. Since sinks are created lazily -- in practice during the first
    // frame -- the writer may be mid-frame right now, and HDF5 is not thread safe here.
    hdf5_writer.drain();
    auto &created = h5_files.emplace(name, HDF5Output(top_folder, filename, settings.configuration_json)).first->second;
    created.set_writer(&hdf5_writer);
    return created;
  }

  template <uint dim, typename VectorType> HDF5Output &OutputSession_impl<dim, VectorType>::hdf5()
  {
    if (!use_hdf5) throw std::logic_error("OutputSession::hdf5: HDF5 output is disabled.");
    return h5_files.at(filename_h5);
  }

  template <uint dim, typename VectorType> void OutputSession_impl<dim, VectorType>::flush_frame(const double time)
  {
    for (const auto &[name, csv] : csv_files)
      csv.validate_frame();

    if constexpr (dim > 0) {
      fe_out.flush(time);
      if (pending_raw_potential) {
        potential_fe_out.flush(time);
        pending_raw_potential.reset();
      }
      if (!pending_eom_potentials.empty()) {
        eom_potential_fe_out.flush(time);
        pending_eom_potentials.clear();
      }
      for (const auto &name : pending_named_fe_outs)
        named_fe_outs.at(name).flush(time);
    }
    {
      ScopedTimer csv_timer(current_frame.csv);
      for (auto &[name, csv] : csv_files)
        csv.flush(time);
    }
    for (auto &[name, hdf] : h5_files)
      hdf.flush(time);

    // Drain the sinks' buckets into this frame. Each take_frame_timings() resets, so nothing is
    // counted twice even though FEOutput already folded in the HDF5Output cost it triggered.
    if constexpr (dim > 0) {
      current_frame += fe_out.take_frame_timings();
      current_frame += potential_fe_out.take_frame_timings();
      current_frame += eom_potential_fe_out.take_frame_timings();
      for (const auto &name : pending_named_fe_outs)
        current_frame += named_fe_outs.at(name).take_frame_timings();
      pending_named_fe_outs.clear();
    }
    for (auto &[name, hdf] : h5_files)
      current_frame += hdf.take_frame_timings();
  }

  template <uint dim, typename VectorType> void OutputSession_impl<dim, VectorType>::record_frame(const double time)
  {
    timings.add(current_frame);

    if (settings.verbosity < 3) return;

    run_reporter.port().debug("output frame t = {:.6e}: total {:.4f}s (contributor {:.4f}s, of which {} potential "
                              "solve(s) {:.4f}s; build_patches {:.4f}s, filter {:.4f}s, hdf5 write {:.4f}s, hdf5 "
                              "open/close {:.4f}s in {} open(s), fe attach {:.4f}s, fe flush {:.4f}s, csv {:.4f}s)",
                              time, current_frame.total, current_frame.contributor, current_frame.potential_solve_count,
                              current_frame.potential_solves, current_frame.build_patches, current_frame.data_filter,
                              current_frame.hdf5_write, current_frame.hdf5_open_close, current_frame.hdf5_open_count,
                              current_frame.fe_attach, current_frame.fe_flush, current_frame.csv);

    diagnostics_port.record("output_timings.csv", time,
                            {{"total", current_frame.total},
                             {"contributor", current_frame.contributor},
                             {"potential_solves", current_frame.potential_solves},
                             {"potential_solve_count", static_cast<double>(current_frame.potential_solve_count)},
                             {"build_patches", current_frame.build_patches},
                             {"data_filter", current_frame.data_filter},
                             {"hdf5_write", current_frame.hdf5_write},
                             {"hdf5_open_close", current_frame.hdf5_open_close},
                             {"hdf5_open_count", static_cast<double>(current_frame.hdf5_open_count)},
                             {"fe_attach", current_frame.fe_attach},
                             {"fe_flush", current_frame.fe_flush},
                             {"csv", current_frame.csv}});
  }

  template <uint dim, typename VectorType>
  void OutputSession_impl<dim, VectorType>::dump_to_csv(const std::string &name,
                                                   const std::vector<std::vector<double>> &values, const bool append,
                                                   const std::vector<std::string> &header)
  {
    const auto path = output_path.resolve(OutputPath::checked_relative(name, "table path"));
    create_folder(path.parent_path().string());
    std::ofstream stream(path, append ? std::ofstream::app : std::ofstream::trunc);
    if (!stream) throw std::runtime_error("OutputSession::dump_table: could not open '" + path.string() + "'.");
    stream << std::scientific;
    if (!append && !header.empty()) {
      for (uint i = 0; i < header.size(); ++i) {
        stream << strip_name(header[i]);
        if (i + 1 != header.size()) stream << ',';
      }
      stream << '\n';
    }
    for (const auto &row : values) {
      for (uint i = 0; i < row.size(); ++i) {
        stream << row[i];
        if (i + 1 != row.size()) stream << ',';
      }
      stream << '\n';
    }
    if (!stream) throw std::runtime_error("OutputSession::dump_table: write failed for '" + path.string() + "'.");
  }

  template <uint dim, typename VectorType> void OutputSession_impl<dim, VectorType>::set_Lambda(const double Lambda)
  {
    this->Lambda = Lambda;
    for (auto &[name, csv] : csv_files)
      csv.set_Lambda(Lambda);
  }

  template <uint dim, typename VectorType> void OutputSession_impl<dim, VectorType>::rethrow_deferred_error()
  {
    if (!deferred_error) return;
    auto error = deferred_error;
    deferred_error = nullptr;
    std::rethrow_exception(error);
  }

  template <uint dim, typename VectorType> void OutputSession_impl<dim, VectorType>::drain()
  {
    std::scoped_lock lock(submission_mutex);
    rethrow_deferred_error();
    if (terminal_error) std::rethrow_exception(terminal_error);
    if (finished || !active) return;

    std::exception_ptr drain_error;
    try {
      fe_out.drain();
    } catch (...) {
      drain_error = std::current_exception();
    }
    try {
      potential_fe_out.drain();
    } catch (...) {
      if (!drain_error) drain_error = std::current_exception();
    }
    try {
      eom_potential_fe_out.drain();
    } catch (...) {
      if (!drain_error) drain_error = std::current_exception();
    }
    for (auto &[name, output] : named_fe_outs) {
      try {
        output.drain();
      } catch (...) {
        if (!drain_error) drain_error = std::current_exception();
      }
    }
    try {
      hdf5_writer.drain();
    } catch (...) {
      if (!drain_error) drain_error = std::current_exception();
    }
    if (drain_error) {
      terminal_error = drain_error;
      std::rethrow_exception(drain_error);
    }
  }

  template <uint dim, typename VectorType> void OutputSession_impl<dim, VectorType>::finish_impl(const bool crashed)
  {
    std::scoped_lock lock(submission_mutex);
    if (finished) {
      rethrow_deferred_error();
      if (terminal_error) std::rethrow_exception(terminal_error);
      return;
    }
    finished = true;
    if (active) {
      try {
        fe_out.finish();
      } catch (...) {
        if (!terminal_error) terminal_error = std::current_exception();
      }
      try {
        potential_fe_out.finish();
      } catch (...) {
        if (!terminal_error) terminal_error = std::current_exception();
      }
      try {
        eom_potential_fe_out.finish();
      } catch (...) {
        if (!terminal_error) terminal_error = std::current_exception();
      }
      for (auto &[name, output] : named_fe_outs) {
        try {
          output.finish();
        } catch (...) {
          if (!terminal_error) terminal_error = std::current_exception();
        }
      }
      // Must happen before h5_files is destroyed: ~HDF5Output closes the file, and that call
      // may not race the worker.
      try {
        hdf5_writer.finish();
      } catch (...) {
        if (!terminal_error) terminal_error = std::current_exception();
      }

      // After the writer is joined, so the mark cannot land before the last frame it describes.
      for (auto &[name, hdf] : h5_files) {
        try {
          hdf.mark_finished(crashed);
        } catch (...) {
          if (!terminal_error) terminal_error = std::current_exception();
        }
      }

      // Reported only after the writer has been joined, so its totals are complete. Always
      // written to the run log; echoed to the console only at verbosity 2+, since the split is
      // a diagnostic rather than something every run needs to scroll past.
      timings.set_writer_totals(hdf5_writer.worker_totals(), hdf5_writer.asynchronous());
      const std::string report = timings.format();
      if (!report.empty()) run_reporter.port().debug(report);
    }
    run_reporter.finish();
    rethrow_deferred_error();
    if (terminal_error) std::rethrow_exception(terminal_error);
  }

  template class OutputSession_impl<0, dealii::Vector<double>>;
  template class OutputSession_impl<1, dealii::Vector<double>>;
  template class OutputSession_impl<2, dealii::Vector<double>>;
  template class OutputSession_impl<3, dealii::Vector<double>>;
  template class OutputSession_impl<1, dealii::BlockVector<double>>;
  template class OutputSession_impl<2, dealii::BlockVector<double>>;
  template class OutputSession_impl<3, dealii::BlockVector<double>>;

#ifdef DEAL_II_WITH_PETSC
  template class OutputSession_impl<0, dealii::PETScWrappers::MPI::Vector>;
  template class OutputSession_impl<1, dealii::PETScWrappers::MPI::Vector>;
  template class OutputSession_impl<2, dealii::PETScWrappers::MPI::Vector>;
  template class OutputSession_impl<3, dealii::PETScWrappers::MPI::Vector>;
#endif
} // namespace DiFfRG
