#include <DiFfRG/discretization/data/output_timings.hh>

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace DiFfRG
{
  FrameTimings &FrameTimings::operator+=(const FrameTimings &other)
  {
    contributor += other.contributor;
    potential_solves += other.potential_solves;
    fe_attach += other.fe_attach;
    build_patches += other.build_patches;
    data_filter += other.data_filter;
    hdf5_write += other.hdf5_write;
    hdf5_open_close += other.hdf5_open_close;
    hdf5_queue_wait += other.hdf5_queue_wait;
    fe_flush += other.fe_flush;
    csv += other.csv;
    total += other.total;
    potential_solve_count += other.potential_solve_count;
    hdf5_open_count += other.hdf5_open_count;
    return *this;
  }

  void OutputTimings::add(const FrameTimings &frame)
  {
    ++n_frames;
    sum += frame;

    max.contributor = std::max(max.contributor, frame.contributor);
    max.potential_solves = std::max(max.potential_solves, frame.potential_solves);
    max.fe_attach = std::max(max.fe_attach, frame.fe_attach);
    max.build_patches = std::max(max.build_patches, frame.build_patches);
    max.data_filter = std::max(max.data_filter, frame.data_filter);
    max.hdf5_write = std::max(max.hdf5_write, frame.hdf5_write);
    max.hdf5_open_close = std::max(max.hdf5_open_close, frame.hdf5_open_close);
    max.hdf5_queue_wait = std::max(max.hdf5_queue_wait, frame.hdf5_queue_wait);
    max.fe_flush = std::max(max.fe_flush, frame.fe_flush);
    max.csv = std::max(max.csv, frame.csv);
    max.total = std::max(max.total, frame.total);
    max.potential_solve_count = std::max(max.potential_solve_count, frame.potential_solve_count);
    max.hdf5_open_count = std::max(max.hdf5_open_count, frame.hdf5_open_count);
  }

  void OutputTimings::set_writer_totals(const FrameTimings &totals, const bool asynchronous)
  {
    writer = totals;
    writer_asynchronous = asynchronous;
  }

  std::string OutputTimings::format() const
  {
    if (n_frames == 0) return {};

    const double frames = static_cast<double>(n_frames);
    // Percentages are of the frame total. They do not add up to 100%: the indented rows are
    // contained in the row above them, and the VTU write itself happens on a worker thread and
    // is deliberately not charged to the frame at all.
    const double reference = sum.total > 0. ? sum.total : 1.;

    std::ostringstream out;
    out << std::fixed;
    out << "Output timings over " << n_frames << " frame" << (n_frames == 1 ? "" : "s") << ":";

    const auto row = [&](const char *name, const double total, const double maximum) {
      out << "\n  " << std::left << std::setw(20) << name << std::right                  //
          << " total " << std::setw(9) << std::setprecision(3) << total << "s"           //
          << " | mean " << std::setw(9) << std::setprecision(4) << total / frames << "s" //
          << " | max " << std::setw(9) << std::setprecision(4) << maximum << "s"         //
          << " | " << std::setw(5) << std::setprecision(1) << 100. * total / reference << "%";
    };

    // Indentation marks containment: an indented row is part of the row above it.
    row("frame total", sum.total, max.total);
    row("  contributor", sum.contributor, max.contributor);
    row("    potential solves", sum.potential_solves, max.potential_solves);
    row("    fe attach", sum.fe_attach, max.fe_attach);
    row("  build_patches", sum.build_patches, max.build_patches);
    row("  data filter", sum.data_filter, max.data_filter);
    row("  hdf5 write", sum.hdf5_write, max.hdf5_write);
    row("  hdf5 open/close", sum.hdf5_open_close, max.hdf5_open_close);
    row("  hdf5 queue wait", sum.hdf5_queue_wait, max.hdf5_queue_wait);
    row("  fe flush", sum.fe_flush, max.fe_flush);
    row("  csv", sum.csv, max.csv);

    if (writer_asynchronous) {
      out << "\n  " << std::left << std::setw(20) << "writer thread" << std::right << " total " << std::setw(9)
          << std::setprecision(3) << writer.hdf5_write + writer.hdf5_open_close << "s | off critical path";
      out << "\n    write " << std::setprecision(3) << writer.hdf5_write << "s | open/close " << writer.hdf5_open_close
          << "s | excluded from frame total";
    }

    out << "\n  " << std::left << std::setw(20) << "counts" << std::right //
        << " potential solves " << sum.potential_solve_count              //
        << " (" << std::setprecision(1) << sum.potential_solve_count / frames
        << "/frame)"
        // Includes the writer thread's opens, which are the bulk of them once it is running.
        << ", hdf5 opens " << sum.hdf5_open_count + writer.hdf5_open_count //
        << " (" << std::setprecision(1) << (sum.hdf5_open_count + writer.hdf5_open_count) / frames << "/frame)";

    return out.str();
  }
} // namespace DiFfRG
