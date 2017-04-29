
#ifndef ICEPACK_OPTIMIZATION_HPP
#define ICEPACK_OPTIMIZATION_HPP

#include <cmath>
#include <cassert>
#include <icepack/numerics/convergence_log.hpp>

namespace icepack
{
  namespace numerics
  {

    /**
     * Find the minimum of a 1D, convex function `f` within an interval `[a, b]`
     * by applying the secant method to its derivative `df`.
     */
    template <typename Functional, typename Derivative>
    double secant_search(
      Functional&& f,
      Derivative&& df,
      const double a,
      const double b,
      const double armijo,
      const double wolfe,
      const size_t level,
      ConvergenceLog& log
    )
    {
      const double f_a = f(a);
      const double df_a = df(a);
      assert(df_a < 0);

      log.add_entry(level, f_a).add_entry(level, f(b));

      double x1 = a;
      double x2 = b;
      double df1 = df_a;
      double df2 = df(x2);

      // These are respectively the Armijo and Wolfe conditions; the Armijo
      // condition states that the objective function has to decrease enough
      // from its starting value, and the second that the derivative of the
      // objective function must be smaller in absolute value than its starting
      // value. See Nocedal and Wright for more explanation.
      while (not ((f(x2) < f_a + armijo * (x2 - a) * df_a)
                  and
                  (std::abs(df2) < wolfe * std::abs(df_a))))
      {
        const double x3 = x2 - df2 * (x2 - x1) / (df2 - df1);

        x1 = x2;
        df1 = df2;

        x2 = x3;
        df2 = df(x2);

        log.add_entry(level, f(x2));
      }

      return x2;
    }


    /**
     * Find the minimum of a 1D, convex function `f` within an interval `[a, b]`
     * by applying the secant method to its derivative `df`.
     */
    template <typename Functional, typename Derivative>
    double secant_search(
      Functional&& f,
      Derivative&& df,
      const double a,
      const double b,
      const double armijo,
      const double wolfe
    )
    {
      ConvergenceLog log;
      return secant_search(f, df, a, b, armijo, wolfe, 0, log);
    }

  } // namespace numerics

} // namespace icepack

#endif
