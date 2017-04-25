
#include <icepack/numerics/convergence_log.hpp>

namespace icepack
{
  namespace numerics
  {
    ConvergenceLog::ConvergenceLog(const std::string& name) :
      method_name_(name), errors_(0)
    {}

    ConvergenceLog::~ConvergenceLog()
    {}

    ConvergenceLog& ConvergenceLog::add_entry(const double error)
    {
      errors_.push_back(error);
      return *this;
    }

    const std::string& ConvergenceLog::method_name() const
    {
      return method_name_;
    }

    const std::vector<double>& ConvergenceLog::errors() const
    {
      return errors_;
    }

  } // namespace numerics
} // namespace icepack
