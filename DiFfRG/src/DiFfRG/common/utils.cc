// standard library
#include <filesystem>
#include <fstream>

// DiFfRG
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  std::string strip_name(const std::string &name)
  {
    std::string stripped_name = name;

    constexpr std::array<char, 32> special_chars{{' ', '^', '$',  '#', '%', '&', '*', '-', '+', '=', '~',
                                                  '`', '!', '@',  '?', '<', '>', ',', '.', ':', ';', '\'',
                                                  '"', '|', '\\', '/', '(', ')', '{', '}', '[', ']'}};
    for (const auto &c : special_chars)
      stripped_name.erase(std::remove(stripped_name.begin(), stripped_name.end(), c), stripped_name.end());
    return stripped_name;
  }

  bool file_exists(const std::string &name)
  {
    std::ifstream f(name.c_str());
    return f.good();
  }

  std::string make_folder(const std::string &path) { return path.back() != '/' ? path + '/' : path; }

  bool create_folder(const std::string &path_)
  {
    auto path = make_folder(path_);
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    return !ec;
  }

  std::string time_format(size_t time_in_seconds)
  {
    using std::to_string;
    if (time_in_seconds < 120) {
      return to_string(time_in_seconds) + "s";
    } else if (time_in_seconds / 60. < 60.) {
      const auto minutes = size_t(time_in_seconds / 60.);
      const auto seconds = size_t(time_in_seconds - minutes * 60);
      return to_string(minutes) + "min" + to_string(seconds) + "s";
    }
    const auto hours = size_t(time_in_seconds / 3600.);
    const auto minutes = size_t(time_in_seconds / 60. - hours * 60);
    const auto seconds = size_t(time_in_seconds - hours * 3600 - minutes * 60);
    return to_string(hours) + "h" + to_string(minutes) + "min" + to_string(seconds) + "s";
  }

  std::string time_format_ms(size_t time_in_miliseconds)
  {
    using std::to_string;
    if (time_in_miliseconds < 1000) {
      return to_string(time_in_miliseconds) + "ms";
    } else if (time_in_miliseconds / 1000. < 60.) {
      const auto seconds = size_t(time_in_miliseconds / 1000.);
      const auto miliseconds = size_t(time_in_miliseconds - seconds * 1000);
      return to_string(seconds) + "s" + to_string(miliseconds) + "ms";
    } else if (time_in_miliseconds / 60000. < 60.) {
      const auto minutes = size_t(time_in_miliseconds / 60000.);
      const auto seconds = size_t(time_in_miliseconds / 1000. - minutes * 60);
      return to_string(minutes) + "min" + to_string(seconds) + "s";
    }
    const auto hours = size_t(time_in_miliseconds / 3600000.);
    const auto minutes = size_t(time_in_miliseconds / 60000. - hours * 60);
    return to_string(hours) + "h" + to_string(minutes) + "min";
  }

  bool has_suffix(const std::string &str, const std::string &suffix)
  {
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
  }
} // namespace DiFfRG
