
#include <icepack/numerics/convergence_log.hpp>
#include "testing.hpp"

int main()
{
  icepack::numerics::ConvergenceLog log;
  CHECK(log.values().size() == 0);

  log.add_entry(0, 1.0).add_entry(1, 0.5).add_entry(0, 0.25);
  CHECK(log.values().size() == 3);
  CHECK_REAL(log.values()[0], 1.0, 1.0e-12);
  CHECK_REAL(log.values()[1], 0.5, 1.0e-12);
  CHECK_REAL(log.values()[2], 0.25, 1.0e-12);
  CHECK(log.levels()[0] == 0);
  CHECK(log.levels()[1] == 1);
  CHECK(log.levels()[2] == 0);

  return 0;
}
