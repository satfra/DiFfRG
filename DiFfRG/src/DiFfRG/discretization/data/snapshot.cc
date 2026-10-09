// DiFfRG
#include <DiFfRG/discretization/data/hdf5_output.hh>
#include <DiFfRG/discretization/data/snapshot.hh>

#include <hdf5lib/hdf5.hh>

// standard library
#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>

namespace DiFfRG
{
  namespace
  {
    double number_from_json(const json::value &v, const std::string &context)
    {
      switch (v.kind()) {
      case json::kind::double_:
        return v.get_double();
      case json::kind::int64:
        return static_cast<double>(v.get_int64());
      case json::kind::uint64:
        return static_cast<double>(v.get_uint64());
      default:
        throw std::runtime_error(context + ": expected a number, got " + json::serialize(v) + ".");
      }
    }

    void write_doubles(hdf5::Group &group, const std::string &name, const std::vector<double> &values)
    {
      // Zero-sized datasets are legal HDF5, but some readers choke on them; an absent dataset reads
      // back as empty just as well.
      if (values.empty()) return;
      group.create_dataset(name, hdf5::type_of<double>(), hdf5::Dataspace::simple({values.size()})).write(values);
    }

    std::vector<double> read_doubles(hdf5::Group &group, const std::string &name)
    {
      if (!group.has_dataset(name)) return {};
      auto dataset = group.open_dataset(name);
      std::vector<double> values(dataset.dataspace().size());
      dataset.read(values);
      return values;
    }

    void write_strings(hdf5::Group &group, const std::string &name, const std::vector<std::string> &values)
    {
      if (values.empty()) return;
      group.create_dataset(name, hdf5::type_of<std::string>(), hdf5::Dataspace::simple({values.size()})).write(values);
    }

    std::vector<std::string> read_strings(hdf5::Group &group, const std::string &name)
    {
      if (!group.has_dataset(name)) return {};
      auto dataset = group.open_dataset(name);
      std::vector<std::string> values(dataset.dataspace().size());
      dataset.read(values);
      return values;
    }

    template <typename T>
    T read_required_attribute(const hdf5::Group &group, const std::string &name, const std::filesystem::path &path)
    {
      if (!group.has_attribute(name))
        throw std::runtime_error("Snapshot '" + path.string() + "' lacks the attribute '" + name +
                                 "'; it is not a DiFfRG snapshot or is corrupt.");
      return group.read_attribute<T>(name);
    }

    /// Open @p path for reading and check that it is a snapshot this DiFfRG can read.
    hdf5::File open_snapshot(const std::filesystem::path &path)
    {
      if (!std::filesystem::exists(path))
        throw std::runtime_error("The snapshot file '" + path.string() + "' does not exist.");
      auto file = hdf5::File::open(path.string(), hdf5::Access::ReadOnly);
      const int version = read_required_attribute<int>(file.root(), "format_version", path);
      if (version != SnapshotData::format_version)
        throw std::runtime_error("Snapshot '" + path.string() + "' has format version " + std::to_string(version) +
                                 ", but this DiFfRG reads version " + std::to_string(SnapshotData::format_version) +
                                 ".");
      return file;
    }

    /// Numbers in their shortest round-trip form (0.1, not boost::json's 1E-1); everything else as JSON.
    std::string format_json(const json::value &v)
    {
      if (v.is_double()) {
        char buffer[64];
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), v.get_double());
        return std::string(buffer, result.ptr);
      }
      return json::serialize(v);
    }

    bool is_ignored(const std::string &pointer, const std::vector<std::string> &ignored_prefixes)
    {
      for (const auto &prefix : ignored_prefixes)
        if (pointer == prefix || pointer.starts_with(prefix + "/")) return true;
      return false;
    }

    /// JSON pointer escaping of an object key (RFC 6901).
    std::string escape_pointer_token(const std::string_view key)
    {
      std::string token;
      for (const char c : key) {
        if (c == '~')
          token += "~0";
        else if (c == '/')
          token += "~1";
        else
          token += c;
      }
      return token;
    }

    void diff_recursive(const json::value &before, const json::value &after, const std::string &pointer,
                        const std::vector<std::string> &ignored_prefixes, std::vector<std::string> &out)
    {
      if (is_ignored(pointer, ignored_prefixes)) return;
      if (before.is_object() && after.is_object()) {
        const auto &b = before.get_object();
        const auto &a = after.get_object();
        for (const auto &entry : b) {
          const std::string child = pointer + "/" + escape_pointer_token(entry.key());
          if (const auto *other = a.if_contains(entry.key()))
            diff_recursive(entry.value(), *other, child, ignored_prefixes, out);
          else if (!is_ignored(child, ignored_prefixes))
            out.push_back(child + ": removed (" + format_json(entry.value()) + ")");
        }
        for (const auto &entry : a) {
          const std::string child = pointer + "/" + escape_pointer_token(entry.key());
          if (!b.contains(entry.key()) && !is_ignored(child, ignored_prefixes))
            out.push_back(child + ": added (" + format_json(entry.value()) + ")");
        }
        return;
      }
      // Numbers compare by value, so that a parameter file saying 2 and one saying 2.0 agree.
      if (before.is_number() && after.is_number()) {
        if (number_from_json(before, pointer) == number_from_json(after, pointer)) return;
      } else if (before == after)
        return;
      out.push_back((pointer.empty() ? std::string("/") : pointer) + ": " + format_json(before) + " -> " +
                    format_json(after));
    }
  } // namespace

  void write_snapshot(const std::filesystem::path &path, const SnapshotData &data)
  {
    if (std::filesystem::exists(path))
      throw std::runtime_error("The snapshot '" + path.string() + "' already exists; snapshots are never overwritten.");
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    auto tmp_path = path;
    tmp_path += ".tmp";

    {
      auto file = hdf5::File::open(tmp_path.string(), hdf5::Access::Truncate);
      auto root = file.root();
      root.write_attribute("format_version", SnapshotData::format_version);
      root.write_attribute("t", data.t);
      root.write_attribute("k", data.k);
      root.write_attribute("Lambda", data.Lambda);
      root.write_attribute("dim", data.dim);
      root.write_attribute("last_adaptation_time", data.last_adaptation_time);

      auto mesh = root.create_group("mesh");
      mesh.write_attribute("fe_name", data.spatial.fe_name);
      mesh.write_attribute("dofs_per_cell", data.spatial.dofs_per_cell);
      mesh.write_attribute("n_coarse_cells", data.spatial.n_coarse_cells);
      mesh.write_attribute("n_dofs", data.spatial.n_dofs);
      write_strings(mesh, "active_cells", data.spatial.active_cells);

      auto state = root.create_group("state");
      state.write_attribute("n_variables", static_cast<unsigned long long>(data.variables.size()));
      write_doubles(state, "spatial", data.spatial.values);
      write_doubles(state, "variables", data.variables);

      auto model = root.create_group("model");
      for (const auto &[name, values] : data.model.data())
        model.create_dataset(name, hdf5::type_of<double>(), hdf5::Dataspace::simple({values.size()})).write(values);

      // The configuration twice: verbatim, which is what a restart diffs against, and as a
      // browsable tree, the same way the run's own .h5 file records it.
      root.create_dataset("config_json", hdf5::type_of<std::string>(), hdf5::Dataspace::scalar())
          .write(data.config_json);
      auto config = root.create_group("config");
      write_config_tree(config, json::parse(data.config_json));

      file.flush();
    }
    std::filesystem::rename(tmp_path, path);
  }

  SnapshotData read_snapshot(const std::filesystem::path &path)
  {
    auto file = open_snapshot(path);
    auto root = file.root();

    SnapshotData data;
    data.t = read_required_attribute<double>(root, "t", path);
    data.k = read_required_attribute<double>(root, "k", path);
    data.Lambda = read_required_attribute<double>(root, "Lambda", path);
    data.dim = read_required_attribute<int>(root, "dim", path);
    data.last_adaptation_time = read_required_attribute<double>(root, "last_adaptation_time", path);

    auto mesh = root.open_group("mesh");
    data.spatial.fe_name = read_required_attribute<std::string>(mesh, "fe_name", path);
    data.spatial.dofs_per_cell = read_required_attribute<unsigned int>(mesh, "dofs_per_cell", path);
    data.spatial.n_coarse_cells = read_required_attribute<unsigned int>(mesh, "n_coarse_cells", path);
    data.spatial.n_dofs = read_required_attribute<unsigned long long>(mesh, "n_dofs", path);
    data.spatial.active_cells = read_strings(mesh, "active_cells");

    auto state = root.open_group("state");
    data.spatial.values = read_doubles(state, "spatial");
    data.variables = read_doubles(state, "variables");
    const auto n_variables = read_required_attribute<unsigned long long>(state, "n_variables", path);
    if (data.variables.size() != n_variables)
      throw std::runtime_error("Snapshot '" + path.string() + "' is corrupt: it declares " +
                               std::to_string(n_variables) + " variables but stores " +
                               std::to_string(data.variables.size()) + ".");

    auto model = root.open_group("model");
    for (const auto &name : model.child_names())
      data.model.set(name, read_doubles(model, name));

    root.open_dataset("config_json").read(data.config_json);

    return data;
  }

  json::value read_snapshot_config(const std::filesystem::path &path)
  {
    auto file = open_snapshot(path);
    std::string config_json;
    file.root().open_dataset("config_json").read(config_json);
    return json::parse(config_json);
  }

  std::filesystem::path snapshot_file(const OutputPath &path, const unsigned int index)
  {
    std::string digits = std::to_string(index);
    if (digits.size() < 3) digits.insert(0, 3 - digits.size(), '0');
    return path.run_file("_snapshot_" + digits, ".h5");
  }

  std::optional<unsigned int> own_snapshot_index(const OutputPath &path, const std::filesystem::path &file)
  {
    std::error_code error;
    const auto directory = std::filesystem::weakly_canonical(std::filesystem::absolute(file).parent_path(), error);
    if (error || directory != std::filesystem::weakly_canonical(path.root(), error) || error) return std::nullopt;

    const std::string name = file.filename().string();
    const std::string prefix = path.run_name() + "_snapshot_";
    const std::string extension = ".h5";
    if (name.size() <= prefix.size() + extension.size() || !name.starts_with(prefix) || !name.ends_with(extension))
      return std::nullopt;
    const std::string digits = name.substr(prefix.size(), name.size() - prefix.size() - extension.size());
    unsigned int index = 0;
    const auto [end, status] = std::from_chars(digits.data(), digits.data() + digits.size(), index);
    if (status != std::errc() || end != digits.data() + digits.size()) return std::nullopt;
    // Only the exact spelling snapshot_file() produces, so that e.g. _snapshot_7 is not taken for _snapshot_007.
    if (snapshot_file(path, index).filename() != file.filename()) return std::nullopt;
    return index;
  }

  std::vector<std::string> config_diff(const json::value &before, const json::value &after,
                                       const std::vector<std::string> &ignored_prefixes)
  {
    std::vector<std::string> out;
    diff_recursive(before, after, "", ignored_prefixes, out);
    return out;
  }

  std::vector<double> config_number_list(const ConfigTree &config, const std::string &key)
  {
    if (!config.contains(key)) return {};
    const json::value root = config;
    const auto &entry = root.at_pointer(key);
    std::vector<double> values;
    if (entry.is_number()) {
      values.push_back(number_from_json(entry, key));
      return values;
    }
    if (!entry.is_array())
      throw std::runtime_error("The configuration entry '" + key + "' must be a number or a list of numbers.");
    for (const auto &element : entry.get_array())
      values.push_back(number_from_json(element, key));
    return values;
  }

  SnapshotSchedule::SnapshotSchedule(const ConfigTree &config, const double Lambda, const double output_dt)
      : output_dt(output_dt)
  {
    snap_to_output_grid = config.get_bool("/timestepping/snapshots/snap_to_output_grid", true);
    stop_after_last_snapshot = config.get_bool("/timestepping/snapshots/stop_after_last", false);

    requested = config_number_list(config, "/timestepping/snapshots/t");
    const auto k_values = config_number_list(config, "/timestepping/snapshots/k");
    if (!k_values.empty() && !(Lambda > 0.))
      throw std::runtime_error("/timestepping/snapshots/k needs a positive /physical/Lambda to convert k into the RG "
                               "time t = ln(Lambda/k); use /timestepping/snapshots/t instead.");
    for (const double k : k_values) {
      if (!(k > 0.)) throw std::runtime_error("/timestepping/snapshots/k: every scale must be positive.");
      requested.push_back(std::log(Lambda / k));
    }
    if (!requested.empty() && snap_to_output_grid && !(output_dt > 0.))
      throw std::runtime_error("Snapping snapshot times to the output grid needs a positive /timestepping/output_dt.");
  }

  std::vector<double> SnapshotSchedule::times_in(const double t_start, const double t_stop) const
  {
    const double eps = 1e-10 * std::max(1., std::abs(t_stop - t_start));
    std::vector<double> times;
    for (const double t : requested) {
      const double snapped = snap_to_output_grid ? t_start + std::round((t - t_start) / output_dt) * output_dt : t;
      if (snapped > t_start + eps && snapped <= t_stop + eps) times.push_back(std::min(snapped, t_stop));
    }
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end(), [&](double a, double b) { return std::abs(a - b) <= eps; }),
                times.end());
    return times;
  }
} // namespace DiFfRG
