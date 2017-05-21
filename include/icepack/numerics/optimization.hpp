
#ifndef ICEPACK_OPTIMIZATION_HPP
#define ICEPACK_OPTIMIZATION_HPP

#include <cmath>
#include <cassert>
#include <limits>
#include <functional>

namespace icepack
{
  namespace numerics
  {
    /**
     * This struct is used for storing options for line searches, such as the
     * Armijo and Wolfe parameters; these determine when a line search has
     * reduced the value and the gradient of the objective function enough to
     * terminate the line search.
     */
    template <typename T>
    struct LineSearchOptions
    {
      static constexpr double armijo_default = 0.01;
      static constexpr double wolfe_default = 0.1;

      using Callback = std::function<void(const double, const T&)>;

      LineSearchOptions(
        const double armijo = armijo_default,
        const double wolfe = wolfe_default,
        const Callback callback = [](const double, const T&){}
      );

      double armijo;
      double wolfe;
      Callback callback;
    };


    template <typename T>
    LineSearchOptions<T>::
    LineSearchOptions(
      const double armijo_,
      const double wolfe_,
      const Callback callback_
    ) : armijo(armijo_), wolfe(wolfe_), callback(callback_)
    {}


    /**
     * Find the minimum of a 1D, convex function `f` within an interval `[a, b]`
     * by applying the secant method to its derivative `df`.
     */
    template <typename T, typename Functional, typename Derivative>
    double line_search(
      const Functional& F,
      const Derivative& dF,
      const T& x0,
      const T& p,
      const double endpoint,
      const LineSearchOptions<T> options = LineSearchOptions<T>()
    )
    {
      const double f0 = F(x0);
      const double df0 = dF(x0, p);
      assert(df0 < 0);

      options.callback(f0, x0);

      double a1 = 0.0;
      double df1 = df0;

      double a2 = endpoint;
      T x = x0 + a2 * p;
      double f2 = F(x);
      double df2 = dF(x, p);

      options.callback(f2, x);

      while (not ((f2 < f0 + options.armijo * a2 * df0)
                  and
                  (std::abs(df2) < options.wolfe * std::abs(df0))))
      {
        const double a3 = a2 - df2 * (a2 - a1) / (df2 - df1);

        a1 = a2;
        df1 = df2;

        a2 = a3;
        x = x0 + a2 * p;
        f2 = F(x);
        df2 = dF(x, p);

        options.callback(f2, x);
      }

      return a2;
    }


    template <typename T>
    struct NewtonSearchOptions
    {
      using Callback = std::function<void(const double, const T&)>;

      NewtonSearchOptions(
        const Callback callback = [](const double, const T&){},
        const LineSearchOptions<T> line_search_options = LineSearchOptions<T>()
      );

      Callback callback;
      LineSearchOptions<T> line_search_options;
    };

    template <typename T>
    NewtonSearchOptions<T>::NewtonSearchOptions(
      const Callback callback_,
      const LineSearchOptions<T> line_search_options_
    ) : callback(callback_), line_search_options(line_search_options_)
    {}


    template
      <typename T, typename Functional, typename Derivative, typename Direction>
    T newton_search(
      const T& x0,
      const Functional& F,
      const Derivative& dF,
      const Direction& P,
      const double tolerance,
      const NewtonSearchOptions<T> options = NewtonSearchOptions<T>()
    )
    {
      T x(x0);
      T p(x0);

      double f0 = std::numeric_limits<double>::infinity();
      double f = F(x);

      options.callback(f, x);

      while (f0 - f > tolerance)
      {
        p = P(x);

        const double alpha =
          line_search(F, dF, x, p, 1.0, options.line_search_options);
        f0 = f;
        x += alpha * p;
        f = F(x);

        options.callback(f, x);
      }

      return x;
    }

  } // namespace numerics

} // namespace icepack

#endif
