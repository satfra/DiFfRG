#pragma once

// DiFfRG
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/discretization/common/snapshot_state.hh>
#include <DiFfRG/discretization/data/output_path.hh>

// standard library
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief Everything needed to continue a flow from an intermediate RG time.
   *
   * Written by AbstractTimestepper::run at the times configured in /timestepping/snapshots and read
   * back when /restart/file is set. See documentation/getting_started/snapshots.md for the use
   * cases and, importantly, for when continuing a flow with a changed configuration is physically
   * meaningful.
   */
  struct SnapshotData {
    static constexpr int format_version = 1;

    /// RG time of the state.
    double t = 0.;
    /// Λ·exp(-t), or NaN when /physical/Lambda was not configured.
    double k = std::numeric_limits<double>::quiet_NaN();
    double Lambda = -1.;
    int dim = 0;
    SnapshotSpatialState spatial;
    std::vector<double> variables;
    ModelState model;
    /// Time of the last mesh adaptation, NaN if the adaptor does not track one.
    double last_adaptation_time = std::numeric_limits<double>::quiet_NaN();
    /// The configuration of the run that wrote the snapshot, verbatim.
    std::string config_json = "{}";
  };

  /**
   * @brief Write @p data to @p path.
   *
   * The file is first written as `<path>.tmp` and then renamed, so a reader never sees a partially
   * written snapshot -- which matters when many restarted runs pick up the snapshot of a seed run
   * that is still going.
   *
   * @throws std::runtime_error if @p path already exists: snapshots are never overwritten.
   */
  void write_snapshot(const std::filesystem::path &path, const SnapshotData &data);

  /** @brief Read a snapshot written by write_snapshot. */
  SnapshotData read_snapshot(const std::filesystem::path &path);

  /** @brief Read only the configuration a snapshot was written with. */
  json::value read_snapshot_config(const std::filesystem::path &path);

  /** @brief `<run name>_snapshot_<nnn>.h5`, the @p index-th snapshot of the run at @p path. */
  std::filesystem::path snapshot_file(const OutputPath &path, unsigned int index);

  /**
   * @brief The index of @p file among the snapshots of the run at @p path, if it is one of them.
   *
   * A restart from one of its own run's snapshots is a continuation of that run: it keeps the
   * snapshot schedule and numbers its snapshots on from there.
   */
  std::optional<unsigned int> own_snapshot_index(const OutputPath &path, const std::filesystem::path &file);

  /**
   * @brief List the differences between two configurations, one line per differing leaf.
   *
   * Lines read `/path: old -> new`, `/path: added (new)` or `/path: removed (old)`. Subtrees whose
   * JSON pointer starts with one of @p ignored_prefixes are skipped.
   */
  std::vector<std::string> config_diff(const json::value &before, const json::value &after,
                                       const std::vector<std::string> &ignored_prefixes = {});

  /**
   * @brief The RG times at which a run writes snapshots.
   *
   * Read from
   * - /timestepping/snapshots/k: list of RG scales k (requires /physical/Lambda),
   * - /timestepping/snapshots/t: list of RG times t,
   * - /timestepping/snapshots/snap_to_output_grid: move each time to the nearest output time
   *   t_start + n·output_dt (default true), so that segmenting the flow does not shift the output grid,
   * - /timestepping/snapshots/stop_after_last: end the run right after its last snapshot (default
   *   false). Meant for seed runs that exist only to produce the snapshot.
   */
  class SnapshotSchedule
  {
  public:
    SnapshotSchedule() = default;
    SnapshotSchedule(const ConfigTree &config, double Lambda, double output_dt);

    /**
     * @brief The snapshot times in (t_start, t_stop], sorted and without duplicates.
     *
     * Snapping is relative to @p t_start, the start of the whole run.
     */
    std::vector<double> times_in(double t_start, double t_stop) const;

    bool empty() const { return requested.empty(); }
    bool stop_after_last() const { return stop_after_last_snapshot; }

  private:
    std::vector<double> requested;
    double output_dt = 0.;
    bool snap_to_output_grid = true;
    bool stop_after_last_snapshot = false;
  };

  /** @brief Parse a JSON array of numbers from @p config at @p key; empty if the key is absent. */
  std::vector<double> config_number_list(const ConfigTree &config, const std::string &key);
} // namespace DiFfRG
