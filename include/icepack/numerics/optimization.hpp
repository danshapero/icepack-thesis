
#ifndef ICEPACK_OPTIMIZATION_HPP
#define ICEPACK_OPTIMIZATION_HPP

#include <cmath>
#include <cassert>
#include <limits>

namespace icepack
{
  namespace numerics
  {
    /**
     * Find the minimum of a 1D, convex function `f` within an interval `[a, b]`
     * by applying the secant method to its derivative `df`.
     */
    template <typename T, typename Functional, typename Derivative>
    double secant_search(
      const Functional& F,
      const Derivative& dF,
      const T& x0,
      const T& p,
      const double endpoint,
      const double armijo,
      const double wolfe
    )
    {
      const double f0 = F(x0);
      const double df0 = dF(x0, p);
      assert(df0 < 0);

      double a1 = 0.0;
      double df1 = df0;

      double a2 = endpoint;
      T x = x0 + a2 * p;
      double f2 = F(x);
      double df2 = dF(x, p);

      while (not ((f2 < f0 + armijo * a2 * df0)
                  and
                  (std::abs(df2) < wolfe * std::abs(df0))))
      {
        const double a3 = a2 - df2 * (a2 - a1) / (df2 - df1);

        a1 = a2;
        df1 = df2;

        a2 = a3;
        x = x0 + a2 * p;
        f2 = F(x);
        df2 = dF(x, p);
      }

      return a2;
    }


    template
      <typename T, typename Functional, typename Derivative, typename Direction>
    T newton_search(
      const T& x0,
      const Functional& F,
      const Derivative& dF,
      const Direction& P,
      const double tolerance
    )
    {
      T x(x0);
      T p(x0);

      double f0 = std::numeric_limits<double>::infinity();
      double f = F(x);

      while (f0 - f > tolerance)
      {
        p = P(x);

        const double alpha = secant_search(F, dF, x, p, 1.0, 0.01, 0.1);
        f0 = f;
        x += alpha * p;
        f = F(x);
      }

      return x;
    }

  } // namespace numerics

} // namespace icepack

#endif
