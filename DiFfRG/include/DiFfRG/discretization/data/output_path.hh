#pragma once

#include <DiFfRG/common/config_tree.hh>

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace DiFfRG
{
  enum class TemporaryRetention { remove_on_destruction, keep };

  /** Shared immutable owner and validated filesystem layout for one output run. */
  class OutputPath
  {
  public:
    OutputPath() = delete;
    OutputPath(std::filesystem::path root, std::string run_name, std::filesystem::path field_directory = "output");
    explicit OutputPath(const ConfigTree &config);

    static OutputPath temporary(TemporaryRetention retention = TemporaryRetention::remove_on_destruction,
                                std::string run_name = "output", std::filesystem::path field_directory = "output");

    OutputPath(const OutputPath &) = default;
    OutputPath &operator=(const OutputPath &) = default;
    OutputPath(OutputPath &&) noexcept = default;
    OutputPath &operator=(OutputPath &&) noexcept = default;

    const std::filesystem::path &root() const noexcept { return root_path; }
    const std::string &run_name() const noexcept { return output_name; }
    const std::filesystem::path &field_directory() const noexcept { return fields_path; }

    std::filesystem::path resolve(std::filesystem::path relative) const;
    std::filesystem::path run_file(std::string_view extension) const;
    /** Companion of run_file carrying an extra suffix: <run name><name_suffix><extension>. */
    std::filesystem::path run_file(std::string_view name_suffix, std::string_view extension) const;
    OutputPath child(std::filesystem::path directory, std::string run_name,
                     std::filesystem::path field_directory = "output") const;
    /** Collective over MPI_COMM_WORLD: rank 0 copies, every rank returns once the copy is done and throws if it failed. */
    void copy_tree_from(const std::filesystem::path &source) const;

    static std::filesystem::path checked_relative(std::filesystem::path path, std::string_view kind = "output path");

  private:
    class OutputRoot;
    OutputPath(std::filesystem::path root, std::string run_name, std::filesystem::path field_directory,
               std::shared_ptr<const OutputRoot> output_root);

    std::filesystem::path root_path;
    std::string output_name;
    std::filesystem::path fields_path;
    std::shared_ptr<const OutputRoot> output_root;
  };
} // namespace DiFfRG
