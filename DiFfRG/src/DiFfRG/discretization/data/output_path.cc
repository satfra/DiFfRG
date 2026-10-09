#include <DiFfRG/discretization/data/output_path.hh>

#include <DiFfRG/common/mpi.hh>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <system_error>

namespace DiFfRG
{
  class OutputPath::OutputRoot
  {
  public:
    enum class CleanupPolicy { preserve, remove_on_last_owner };

    OutputRoot(std::filesystem::path root, const CleanupPolicy cleanup) : root(std::move(root)), cleanup(cleanup) {}

    ~OutputRoot() noexcept
    {
      if (cleanup == CleanupPolicy::preserve) return;
      std::error_code error;
      std::filesystem::remove_all(root, error);
    }

  private:
    std::filesystem::path root;
    CleanupPolicy cleanup;
  };

  namespace
  {
    std::filesystem::path normalized_root(const std::filesystem::path &path)
    {
      if (path.empty()) throw std::invalid_argument("OutputPath: output root must not be empty.");
      return std::filesystem::absolute(path).lexically_normal();
    }

    std::string checked_run_name(const std::string &name)
    {
      const auto path = OutputPath::checked_relative(name, "run name");
      if (path.has_parent_path()) throw std::invalid_argument("OutputPath: run name must be a single path element.");
      return path.string();
    }

    std::filesystem::path create_temporary_root()
    {
      static std::atomic<std::uint64_t> sequence{0};
      std::random_device random;
      const auto temporary_root = std::filesystem::temp_directory_path();
      for (unsigned int attempt = 0; attempt < 128; ++attempt) {
        const auto counter = sequence.fetch_add(1, std::memory_order_relaxed);
        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto token = static_cast<std::uint64_t>(random()) ^ static_cast<std::uint64_t>(ticks) ^ counter;
        const auto candidate = temporary_root / ("diffrg-" + std::to_string(token));
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error)) return normalized_root(candidate);
        if (error && error != std::errc::file_exists)
          throw std::runtime_error("OutputPath: could not create temporary directory '" + candidate.string() +
                                   "': " + error.message());
      }
      throw std::runtime_error("OutputPath: could not create a unique temporary directory.");
    }

  } // namespace

  OutputPath::OutputPath(std::filesystem::path root, std::string run_name, std::filesystem::path field_directory)
      : root_path(normalized_root(root)), output_name(checked_run_name(run_name)),
        fields_path(checked_relative(std::move(field_directory), "field directory")),
        output_root(std::make_shared<OutputRoot>(root_path, OutputRoot::CleanupPolicy::preserve))
  {
  }

  OutputPath::OutputPath(const ConfigTree &config)
      : OutputPath(config.get_string("/output/folder", "./"), config.get_string("/output/name", "output"),
                   config.get_string("/output/field_directory", config.get_string("/output/name", "output")))
  {
  }

  OutputPath::OutputPath(std::filesystem::path root, std::string run_name, std::filesystem::path field_directory,
                         std::shared_ptr<const OutputRoot> output_root)
      : root_path(normalized_root(root)), output_name(checked_run_name(run_name)),
        fields_path(checked_relative(std::move(field_directory), "field directory")),
        output_root(std::move(output_root))
  {
  }

  OutputPath OutputPath::temporary(const TemporaryRetention retention, std::string run_name,
                                   std::filesystem::path field_directory)
  {
    auto root = create_temporary_root();
    const auto cleanup = retention == TemporaryRetention::keep ? OutputRoot::CleanupPolicy::preserve
                                                               : OutputRoot::CleanupPolicy::remove_on_last_owner;
    auto owner = std::make_shared<OutputRoot>(root, cleanup);
    if (retention == TemporaryRetention::keep) std::clog << "Temporary output retained at " << root << '\n';
    return OutputPath(std::move(root), std::move(run_name), std::move(field_directory), std::move(owner));
  }

  std::filesystem::path OutputPath::checked_relative(std::filesystem::path path, const std::string_view kind)
  {
    if (path.empty()) throw std::invalid_argument("OutputPath: " + std::string(kind) + " must not be empty.");
    const auto normalized = path.lexically_normal();
    if (path.is_absolute() || normalized.empty() || normalized == ".")
      throw std::invalid_argument("OutputPath: unsafe " + std::string(kind) + " '" + path.string() + "'.");
    for (const auto &part : normalized)
      if (part == "..")
        throw std::invalid_argument("OutputPath: unsafe " + std::string(kind) + " '" + path.string() + "'.");
    return normalized;
  }

  std::filesystem::path OutputPath::resolve(std::filesystem::path relative) const
  {
    return root_path / checked_relative(std::move(relative));
  }

  std::filesystem::path OutputPath::run_file(const std::string_view extension) const
  {
    return run_file(std::string_view{}, extension);
  }

  std::filesystem::path OutputPath::run_file(const std::string_view name_suffix,
                                             const std::string_view extension) const
  {
    if (extension.empty() || extension.front() != '.' || extension.find_first_of("/\\") != std::string_view::npos)
      throw std::invalid_argument("OutputPath: file extension must begin with '.' and contain no path separators.");
    if (name_suffix.find_first_of("/\\") != std::string_view::npos)
      throw std::invalid_argument("OutputPath: run file name suffix must contain no path separators.");
    return root_path / (output_name + std::string(name_suffix) + std::string(extension));
  }

  OutputPath OutputPath::child(std::filesystem::path directory, std::string run_name,
                               std::filesystem::path field_directory) const
  {
    return OutputPath(resolve(std::move(directory)), std::move(run_name), std::move(field_directory), output_root);
  }

  void OutputPath::copy_tree_from(const std::filesystem::path &source) const
  {
    // Collective. Every rank calls this on the same paths, but only rank 0 may copy: two ranks
    // copying onto one tree with overwrite_existing truncate and rewrite the same files, and a rank
    // that finishes first then reads a file the other is still rewriting ("file signature not
    // found" from HDF5). The other ranks wait for the copy and learn its outcome, so a failure
    // throws everywhere rather than hanging the ranks that did not see it.
    std::string error;
    if (MPI::rank(MPI_COMM_WORLD) == 0) {
      if (!std::filesystem::exists(source))
        error = "OutputPath: artifact source does not exist: " + source.string();
      else
        try {
          std::filesystem::create_directories(root_path);
          std::filesystem::copy(source, root_path,
                                std::filesystem::copy_options::recursive |
                                    std::filesystem::copy_options::overwrite_existing);
        } catch (const std::filesystem::filesystem_error &e) {
          error = "OutputPath: could not copy artifact tree from '" + source.string() + "' to '" +
                  root_path.string() + "': " + e.what();
        }
    }
    std::uint64_t length = error.size();
    MPI::bcast(MPI_COMM_WORLD, &length, sizeof(length));
    if (length == 0) return;
    error.resize(length);
    MPI::bcast(MPI_COMM_WORLD, error.data(), length);
    throw std::runtime_error(error);
  }
} // namespace DiFfRG
