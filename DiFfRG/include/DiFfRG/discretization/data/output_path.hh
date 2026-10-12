#pragma once

#include <DiFfRG/common/config_tree.hh>

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace DiFfRG
{
  /// Whether a temporary output folder is deleted when the last OutputPath referring to it is destroyed.
  enum class TemporaryRetention { remove_on_destruction, keep };

  /**
   * @brief Where a run writes its files: a root folder, a run name, and a fields subfolder.
   *
   * Files are named `<root>/<run_name>...`; the per-frame VTK files go to `<root>/<field_directory>/`.
   */
  class OutputPath
  {
  public:
    OutputPath() = delete;
    /**
     * @brief Paths from explicit values.
     *
     * @param root Output folder (made absolute).
     * @param run_name Prefix of all files; a single path component.
     * @param field_directory Folder of the VTK frames, relative to @p root, without "..".
     */
    OutputPath(std::filesystem::path root, std::string run_name, std::filesystem::path field_directory = "output");
    /**
     * @brief Paths from the parameters: `/output/folder` (default "./"), `/output/name` (default "output") and
     * `/output/field_directory` (default: the run name).
     */
    explicit OutputPath(const ConfigTree &config);

    /// A new, uniquely named folder in the system's temporary directory.
    static OutputPath temporary(TemporaryRetention retention = TemporaryRetention::remove_on_destruction,
                                std::string run_name = "output", std::filesystem::path field_directory = "output");

    OutputPath(const OutputPath &) = default;
    OutputPath &operator=(const OutputPath &) = default;
    OutputPath(OutputPath &&) noexcept = default;
    OutputPath &operator=(OutputPath &&) noexcept = default;

    /// The absolute output folder.
    const std::filesystem::path &root() const noexcept { return root_path; }
    /// The run name.
    const std::string &run_name() const noexcept { return output_name; }
    /// The folder of the VTK frames, relative to root().
    const std::filesystem::path &field_directory() const noexcept { return fields_path; }

    /// `root() / relative`; throws if @p relative is absolute or contains "..".
    std::filesystem::path resolve(std::filesystem::path relative) const;
    /// `root() / (run_name() + extension)`, e.g. run_file(".h5").
    std::filesystem::path run_file(std::string_view extension) const;
    /// `root() / (run_name() + name_suffix + extension)`.
    std::filesystem::path run_file(std::string_view name_suffix, std::string_view extension) const;
    /// Paths for another run in the subfolder @p directory of root(), sharing this one's lifetime.
    OutputPath child(std::filesystem::path directory, std::string run_name,
                     std::filesystem::path field_directory = "output") const;
    /// Copy the folder tree @p source into root(). Collective: rank 0 copies, all ranks throw if it failed.
    void copy_tree_from(const std::filesystem::path &source) const;

    /// @p path, normalized; throws if it is empty, absolute or contains "..".
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
