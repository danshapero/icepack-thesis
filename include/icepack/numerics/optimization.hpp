
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
     * @brief Struct that stores various options for line searches
     *
     * The decision for when to stop a line search depends on two parameters.
     *
     * When performing a line search, we would like to guarantee that the
     * objective function has decreased enough; if it hasn't, we should keep
     * searching. We can say that \f$\alpha\f$ is a good enough result for the
     * line search if the objective function has been reduced by some multiple
     * of \f$\alpha\f$ and the rate of change of the objective along the search
     * direction:
     * \f[
     *   f(x + \alpha \cdot p) \le f(x) + c_1\alpha\langle df(x), p\rangle
     * \f]
     * keeping in mind that, since the search direction \f$p\f$ is a descent
     * direction, the \f$\langle df(x), p\rangle < 0\f$. This is the *Armijo*
     * condition, and \f$c_1\f$ is called the Armijo parameter.
     *
     * The second desirable property of a line search is that the derivative of
     * the objective function is closer to 0 than it was before:
     * \f[
     *   |\langle df(x + \alpha\cdot p), p\rangle| \le c_2|\langle df(x), p\rangle|.
     * \f]
     * This is the *Wolfe* condition, and \f$c_2\f$ is the Wolfe parameter.
     *
     * The Armijo and Wolfe parameters dictate how much exactness we use in
     * performing the line search. (Usually one has \f$c_1 < c_2\f$.) The degree
     * to which you use a very accurate line search depends on how good your
     * search direction is and how expensive it is to compute. For more
     * information, see Nocedal and Wright, or the [Wikipedia article]
     * (https://en.wikipedia.org/wiki/Wolfe_conditions).
     *
     * Additionally, this struct stores an optional callback, a function that
     * can be invoked on every iteration of the line search. This is useful for
     * monitoring the convergence of the optimization algorithms in the event
     * that something breaks.
     */
    template <typename T>
    struct LineSearchOptions
    {
      static constexpr double armijo_default = 0.01;
      static constexpr double wolfe_default = 0.1;

      using Callback = std::function<void(const double, const T&)>;

      /**
       * Construct a new set of line search options. The default values of the
       * Armijo and Wolfe parameters are set to the values suggested in Nocedal
       * and Wright for quasi-Newton methods that have accurate approximations
       * of the Hessian matrix.
       */
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
     * @brief Find the minimum of a convex functional along a search direction.
     *
     * This routine requires procedures for evaluating the objective functional
     * and its directional derivative. This procedure is used in (quasi-) Newton
     * methods for optimization.
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


    /**
     * @brief Struct that stores various options Newton search.
     *
     * The possible options include a callback to be invoked at every iteration
     * of the Newton search, and the options passed to the line search along the
     * search direction at every step. The line search options should be chosen
     * based on the method you're using to compute the search direction; you'll
     * want a more accurate line search when using the full Newton update than
     * you would for, say, gradient descent.
     */
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


    /**
     * @brief Find an approximate minimizer of a convex objective functional
     * using a (quasi-) Newton method.
     *
     * This routine requires procedures for evaluating the objective functional,
     * the directional derivative, and computing a search direction. The search
     * direction could be computed using the full Newton update, or some quasi-
     * Newton approximation such as L-BFGS.
     *
     * The procedure for computing the search direction has to return an object
     * with the same type `T` as the domain of the objective functional. For
     * example, the domain type `T` for solving the shallow shelf equations is
     * `VectorField<2>`. Pure gradient descent, which uses the derivative of the
     * objective functional as a search direction, does not meet this criterion;
     * the derivative of the objective is of type `DualVectorField<2>`. To
     * compute a search direction, we need some way of mapping back from
     * dual fields to primal fields. The Newton search direction, which is
     * obtained by mutliplying the derivative by the inverse of the second
     * derivative operator, does exactly that.
     */
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
