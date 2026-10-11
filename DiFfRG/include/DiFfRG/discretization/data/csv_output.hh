#pragma once

// DiFfRG
#include <DiFfRG/common/config_tree.hh>

// standard library
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief One CSV table with one row per output frame.
   *
   * Columns: `t`, then `k [GeV]` \f$= \Lambda e^{-t}\f$ if \f$\Lambda > 0\f$ was set, then the values in the order
   * in which they were added in the first frame. Every frame must add exactly the same names (in any order), each
   * once; otherwise flush() throws. Applications obtain one through OutputFrame::table rather than constructing it.
   */
  class CsvOutput
  {
  public:
    /**
     * @brief Create (truncate) the file @p top_folder / @p output_name, creating folders as needed.
     */
    CsvOutput(const std::string top_folder, const std::string output_name);
    /// The ConfigTree is ignored.
    [[deprecated("Use CsvOutput(top_folder, output_name) instead")]] CsvOutput(
        const std::string top_folder, const std::string output_name, [[maybe_unused]] const ConfigTree &config)
        : CsvOutput(std::move(top_folder), std::move(output_name))
    {
    }

    /// Add the value of column @p name to the current frame; throws if @p name was already added in this frame.
    void value(const std::string &name, const double value);

    /// Write the current frame as one row at RG time @p time (the first call also writes the header).
    void flush(const double time);
    /// Throw if the current frame does not contain exactly the columns of the first frame.
    void validate_frame() const;

    /**
     * @brief Set \f$\Lambda\f$; if \f$\Lambda > 0\f$, a column `k [GeV]` \f$= \Lambda e^{-t}\f$ is added.
     *
     * Must be called before the first flush(); changing it afterwards throws.
     */
    void set_Lambda(const double Lambda);

  private:
    const std::string top_folder;
    const std::string output_name;
    std::ofstream output_stream;

    std::vector<std::string> insertion_order;
    std::unordered_map<std::string, std::vector<double>> values;
    std::unordered_set<std::string> written_this_frame;

    std::vector<std::string> header;
    std::vector<double> time_values;
    std::vector<double> k_values;

    double Lambda;
  };
} // namespace DiFfRG
