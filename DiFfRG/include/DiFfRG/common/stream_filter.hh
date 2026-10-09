#pragma once

// standard libraries
#include <ostream>
#include <streambuf>
#include <string>
#include <string_view>

namespace DiFfRG
{
  /**
   * @brief A std::streambuf that forwards everything except whole lines starting with a prefix.
   *
   * Bytes are withheld only while the current line is still a prefix of the pattern; the moment it
   * diverges, all of them are forwarded at once. Nothing written for another reason is therefore
   * delayed. That property is the point of the class rather than an optimization: it is used around
   * calls that may abort the process, and a line-buffering filter would swallow the dying message.
   */
  class LinePrefixFilter : public std::streambuf
  {
  public:
    LinePrefixFilter(std::streambuf *sink, std::string_view prefix) : sink(sink), prefix(prefix) {}

  protected:
    int_type overflow(int_type ch) override
    {
      if (traits_type::eq_int_type(ch, traits_type::eof())) return traits_type::not_eof(ch);
      const char c = traits_type::to_char_type(ch);

      switch (state) {
      case State::undecided:
        held.push_back(c);
        if (c == '\n') {
          // A complete line that never diverged but never completed the prefix either.
          forward();
          reset();
        } else if (held.size() > prefix.size() || prefix.compare(0, held.size(), held) != 0) {
          forward();
          state = State::forwarding;
          held.clear();
        } else if (held.size() == prefix.size()) {
          state = State::dropping;
          held.clear();
        }
        break;
      case State::forwarding:
        sink->sputc(c);
        if (c == '\n') reset();
        break;
      case State::dropping:
        if (c == '\n') reset();
        break;
      }
      return ch;
    }

    int sync() override
    {
      // Whatever is held belongs to an unfinished line that may never be completed, so let it out.
      if (!held.empty()) {
        forward();
        state = State::forwarding;
        held.clear();
      }
      return sink->pubsync();
    }

  private:
    enum class State {
      undecided,  ///< the line so far is still a prefix of the pattern
      forwarding, ///< this line has diverged and is passed through
      dropping    ///< this line matched and is discarded up to its newline
    };

    void forward() { sink->sputn(held.data(), static_cast<std::streamsize>(held.size())); }
    void reset()
    {
      state = State::undecided;
      held.clear();
    }

    std::streambuf *sink;
    std::string_view prefix;
    std::string held;
    State state = State::undecided;
  };

  /**
   * @brief Installs a LinePrefixFilter on a stream for the duration of a scope.
   */
  class ScopedLineFilter
  {
  public:
    ScopedLineFilter(std::ostream &stream, std::string_view prefix)
        : stream(stream), filter(stream.rdbuf(), prefix), previous(stream.rdbuf(&filter))
    {
    }
    ~ScopedLineFilter()
    {
      stream.flush();
      stream.rdbuf(previous);
    }

    ScopedLineFilter(const ScopedLineFilter &) = delete;
    ScopedLineFilter &operator=(const ScopedLineFilter &) = delete;

  private:
    std::ostream &stream;
    LinePrefixFilter filter;
    std::streambuf *previous;
  };
} // namespace DiFfRG
