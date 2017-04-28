
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
       * Clear all the memory used by a convergence log.
       */
      virtual ~ConvergenceLog();

      /**
       * Add a new entry to the convergence log.
       */
      ConvergenceLog& add_entry(const size_t level, const double value);

      /**
       * Return a reference to the array describing the level of each iteration
       * of the method.
       */
      const std::vector<size_t>& levels() const;

      /**
       * Return a reference to the array of values. Hopefully it's decreasing.
       */
      const std::vector<double>& values() const;

    protected:
      std::vector<size_t> levels_;
      std::vector<double> values_;
    };


  } // namespace numerics
} // namespace icepack

#endif
