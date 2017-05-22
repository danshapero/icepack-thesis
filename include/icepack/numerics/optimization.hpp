
#ifndef ICEPACK_OPTIMIZATION_HPP
#define ICEPACK_OPTIMIZATION_HPP

#include <cmath>
#include <cassert>
#include <limits>
#include <functional>
#include <icepack/numerics/sequence.hpp>

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
      Sequence<double> as(3), fs(3), dfs(3);

      as[0] = 0.0;
      as[1] = endpoint;
      fs[0] = F(x0);
      dfs[0] = dF(x0, p);
      assert(dfs[0] < 0);
      T x(x0);

      const auto stopping_criterion =
        [=](const double a, const double f, const double df)
        {
          return (f < fs[0] + options.armijo * a * dfs[0]) and
                 (std::abs(df) < options.wolfe * std::abs(dfs[0]));
        };

      size_t n = 0;
      do
      {
        n += 1;
        x = x0 + as[n] * p;
        fs[n] = F(x);
        dfs[n] = dF(x, p);
        options.callback(fs[n], x);

        const double d2f = (dfs[n] - dfs[n - 1]) / (as[n] - as[n - 1]);
        as[n + 1] = as[n] - dfs[n] / d2f;
      } while (not stopping_criterion(as[n], fs[n], dfs[n]));

      return as[n];
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
