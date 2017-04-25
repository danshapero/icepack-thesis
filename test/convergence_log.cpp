
#include <icepack/numerics/convergence_log.hpp>
#include "testing.hpp"

int main()
{
  const std::string name("dummy method");
  icepack::numerics::ConvergenceLog log(name);
  CHECK(log.method_name() == name);
  CHECK(log.errors().size() == 0);

  log.add_entry(1.0).add_entry(0.5).add_entry(0.25);
  CHECK(log.errors().size() == 3);
  CHECK_REAL(log.errors()[0], 1.0, 1.0e-12);
  CHECK_REAL(log.errors()[1], 0.5, 1.0e-12);
  CHECK_REAL(log.errors()[2], 0.25, 1.0e-12);

  return 0;
}
