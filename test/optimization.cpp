
#include <iostream>
#include <icepack/numerics/optimization.hpp>
#include "testing.hpp"

int main()
{
  TEST_SUITE("golden section search")
  {
    const double x0 = 3.0;
    const auto f = [=](const double x) { return std::cosh(x - x0); };

    const double a = 0.0;
    const double b = 10.0;

    const double x = icepack::numerics::golden_section_search(f, a, b, 1.0e-6);
    CHECK_REAL(x, x0, 1.0e-4);
  }

  return 0;
}
