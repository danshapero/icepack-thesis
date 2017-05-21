
#include <iostream>
#include <icepack/numerics/optimization.hpp>
#include "testing.hpp"

int main()
{
  const double x0 = 3.0;
  const auto f = [=](const double x) { return std::cosh(x - x0); };
  const auto df =
    [=](const double x, const double p) { return std::sinh(x - x0) * p; };

  const double a = 0.0;
  const double b = 10.0;

  TEST_SUITE("secant search")
  {
    // Making the constant in the Wolfe condition really small makes the line
    // search pretty much exact. In practice, we might use a Wolfe constant of
    // something like 0.1 for an inexact line search.
    const double wolfe = 1.0e-6;
    const double armijo = 1.0e-2;
    size_t iterations = 0;
    const auto callback =
      [&](const double, const double)
      {
        iterations += 1;
      };

    icepack::numerics::LineSearchOptions<double>
      options{armijo, wolfe, callback};

    const double p = (df(a, 1.0) < 0) ? 1 : -1;
    const double x = icepack::numerics::line_search(f, df, a, p, b, options);
    CHECK_REAL(x, x0, 1.0e-6);
    CHECK(iterations != 0);
  }

  return 0;
}
