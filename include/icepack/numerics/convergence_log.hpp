
#ifndef ICEPACK_CONVERGENCE_LOG_HPP
#define ICEPACK_CONVERGENCE_LOG_HPP

#include <string>
#include <vector>

namespace icepack
{
  namespace numerics
  {
    /**
     * This class is used to keep track of the convergence history of an
     * iterative method. This class is used for several things:
     *   - debugging nonlinear solvers in the event of failure
     *   - checking that nonlinear solvers are achieving the theoretical rate
     *     of convergence, for example that Newton's method is quadratic.
     *   - optimizing the use of multiple nonlinear solvers, for example when
     *     to switching from a low-order method like LBFGS to Newton
     */
    class ConvergenceLog
    {
    public:
      /**
       * Create an empty convergence log. Entries can be added using the method
       * `add_entry` defined below.
       */
      ConvergenceLog(const std::string& name = "");

      /**
       * Clear all the memory used by a convergence log.
       */
      virtual ~ConvergenceLog();

      /**
       * Add a new entry to the convergence log.
       */
      virtual ConvergenceLog& add_entry(const double error);

      /**
       * Return the name of the iterative method that this convergence log was
       * for, i.e. "Newton", "L-BFGS", etc.
       */
      const std::string& method_name() const;

      /**
       * Return a reference to the array of errors. Hopefully it's decreasing.
       */
      const std::vector<double>& errors() const;

    protected:
      std::string method_name_;
      std::vector<double> errors_;
    };


  } // namespace numerics
} // namespace icepack

#endif
