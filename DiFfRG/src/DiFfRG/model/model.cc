// DiFfRG
#include "DiFfRG/common/config_tree.hh"
#include <DiFfRG/common/math.hh>
#include <DiFfRG/model/model.hh>

namespace DiFfRG
{
  namespace def
  {
    void Time::set_time(double t_) { t = t_; }
    const double &Time::get_time() const { return t; }

    fRG::fRG(double Lambda) : Lambda(Lambda), k(Lambda) { set_time(0.); }

    fRG::fRG(const ConfigTree &config) : Lambda(config.get_double_or_warn("/physical/Lambda", 1.0)), k(Lambda)
    {
      set_time(0.);
    }

    void fRG::set_time(double t_)
    {
      if (time_initialized && t_ == t) return;
      time_initialized = true;
      t = t_;
      k = std::exp(-static_cast<long double>(t)) * Lambda;
      k2 = powr<2>(k);
      k3 = powr<3>(k);
      k4 = powr<4>(k);
      k5 = powr<5>(k);
      k6 = powr<6>(k);
    }
    const double &fRG::get_time() const { return t; }
  } // namespace def
} // namespace DiFfRG