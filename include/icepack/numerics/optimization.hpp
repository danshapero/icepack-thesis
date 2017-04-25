
#ifndef ICEPACK_OPTIMIZATION_HPP
#define ICEPACK_OPTIMIZATION_HPP

#include <cmath>
#include <cassert>

namespace icepack
{
  namespace numerics
  {
    /**
     * Find the minimum of a 1D, convex function `f` within an interval `[a, b]`
     * using Golden section search.
     */
    template <typename Functional>
    double golden_section_search(
      Functional&& f,
      double a,
      double b,
      const double eps
    )
    {
      assert(b > a);
      const double phi = 0.5 * (std::sqrt(5.0) - 1);
      const double d = b - a;

      while (b - a > eps * d)
      {
        const double L = b - a;
        const double A = a + L * (1 - phi);
        const double B = b - L * (1 - phi);
        const double fA = f(A), fB = f(B);

        /* There are four cases to consider if `f` is convex. In these two
         * cases, the minimum lies in the interval [A, b]:
         *  ^                 ^
         *  |  o              |  o
         *  |                 |          o
         *  |    o            |    o
         *  |      o          |       o
         *  |          o      |
         *  *------------>    *------------>
         *     a A B   b         a A  B  b
         *
         * and in these two cases, the minimum lies in the interval [a, B].
         * ^                  ^
         * |         o        |         o
         * |                  | o
         * |       o          |       o
         * |     o            |    o
         * | o                |
         * *------------->    *------------>
         *   a   A B b          a  A  B  b
         *
         * These cases are characterized by whether f(A) >= f(B) or vice versa.
         */

        if (fA >= fB)
          a = A;
        else
          b = B;
      }

      return (a + b) / 2;
    }

  } // namespace numerics

} // namespace icepack

#endif
