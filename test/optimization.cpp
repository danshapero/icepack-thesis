
#include <iostream>
#include <icepack/numerics/optimization.hpp>
#include "testing.hpp"

int main()
{

  const double x0 = 3.0;
  const auto f = [=](const double x) { return std::cosh(x - x0); };
  const auto df = [=](const double x) { return std::sinh(x - x0); };

  const double a = 0.0;
  const double b = 10.0;


  TEST_SUITE("golden section search")
  {
    const double x = icepack::numerics::golden_section_search(f, a, b, 1.0e-6);
    CHECK_REAL(x, x0, 1.0e-4);
  }


  TEST_SUITE("secant search")
  {
    // Making the constant in the Wolfe condition really small makes the line
    // search pretty much exact. In practice, we might use a Wolfe constant of
    // something like 0.1 for an inexact line search.
    const double wolfe = 1.0e-6;
    const double armijo = 1.0e-2;

    const double x =
      icepack::numerics::secant_search(f, df, a, b, armijo, wolfe);
    CHECK_REAL(x, x0, 1.0e-6);
  }

  return 0;
}
