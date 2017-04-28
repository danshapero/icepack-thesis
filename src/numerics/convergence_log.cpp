
#include <icepack/numerics/convergence_log.hpp>

namespace icepack
{
  namespace numerics
  {
    ConvergenceLog::~ConvergenceLog()
    {}

    ConvergenceLog&
    ConvergenceLog::add_entry(const size_t level, const double value)
    {
      levels_.push_back(level);
      values_.push_back(value);
      return *this;
    }

    const std::vector<size_t>& ConvergenceLog::levels() const
    {
      return levels_;
    }

    const std::vector<double>& ConvergenceLog::values() const
    {
      return values_;
    }

  } // namespace numerics
} // namespace icepack
