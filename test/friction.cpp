
#include <icepack/physics/friction.hpp>
#include "testing.hpp"

using dealii::Tensor;
using dealii::SymmetricTensor;

int main(int argc, char ** argv)
{
  const auto args = icepack::testing::get_cmdline_args(argc, argv);
  const bool verbose = args.count("--verbose");
  const size_t num_samples = 8;

  const auto C = [](const double beta){ return beta * beta; };
  const auto dC = [](const double beta){ return 2 * beta; };
  icepack::BasalStress basal_stress(3.0, C, dC);
  const double m = basal_stress.m;

  TEST_SUITE("derivative of basal stress w.r.t. velocity")
  {
    const double Tau0 = 1.0e-2;
    const double U0 = 1.0e2;
    const double beta = std::sqrt(Tau0 / std::pow(U0, 1/m));

    const Tensor<1, 2> u{{U0, 0.0}};
    const Tensor<1, 2> tau0 = basal_stress(beta, u);
    CHECK_REAL(tau0.norm(), Tau0, 1.0e-8);

    const Tensor<1, 2> du{{0.0, -20.0}};
    const SymmetricTensor<2, 2> dtau = basal_stress.du(beta, u);

    std::vector<double> errors(num_samples);
    for (unsigned int k = 0; k < num_samples; ++k)
    {
      const double delta = 1.0 / std::pow(2.0, k);
      const Tensor<1, 2> tau = basal_stress(beta, u + delta * du);
      const Tensor<1, 2> tau_approx = tau0 + delta * dtau * du;
      errors[k] = (tau - tau_approx).norm();
    }

    if (verbose)
      icepack::testing::print_errors(errors);

    CHECK(icepack::testing::is_decreasing(errors));
  }

  return 0;
}
