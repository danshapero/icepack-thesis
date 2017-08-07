
#include <icepack/physics/viscosity.hpp>
#include "testing.hpp"

using dealii::SymmetricTensor;

int main(int argc, char ** argv)
{
  const auto args = icepack::testing::get_cmdline_args(argc, argv);
  const bool verbose = args.count("--verbose");
  const size_t num_samples = 8;

  icepack::MembraneStress membrane_stress;

  TEST_SUITE("derivative of rheology")
  {
    const auto test = [&](const double T0, const double dT)
    {
      const double B0 = membrane_stress.B(T0);
      const double dB = membrane_stress.dB(T0);

      std::vector<double> errors(num_samples);
      for (unsigned int k = 0; k < num_samples; ++k)
      {
        const double delta = 1.0 / std::pow(2.0, k);
        const double B = membrane_stress.B(T0 + delta * dT);
        const double B_approx = B0 + delta * dB * dT;
        errors[k] = std::abs(B - B_approx);
      }

      if (verbose)
        icepack::testing::print_errors(errors);

      CHECK(icepack::testing::is_decreasing(errors));
    };

    // Check that the derivative of the rheology is implemented right both above
    // and below the transition temp of 263.15K.
    test(253.0, 5.0);
    test(270.0, -5.0);
  }


  TEST_SUITE("derivative of membrane stress w.r.t. temperature")
  {
    const double T0 = 253.0;
    const double dT = 5.0;
    const SymmetricTensor<2, 2> eps{{0.5, -0.8, 2.3}};

    const SymmetricTensor<2, 2> M0 = membrane_stress(T0, eps);
    const SymmetricTensor<2, 2> dM = membrane_stress.dtheta(T0, eps);

    std::vector<double> errors(num_samples);
    for (unsigned int k = 0; k < num_samples; ++k)
    {
      const double delta = 1.0 / std::pow(2.0, k);
      const SymmetricTensor<2, 2> M = membrane_stress(T0 + delta * dT, eps);
      const SymmetricTensor<2, 2> M_approx = M0 + delta * dM * dT;
      errors[k] = (M - M_approx).norm();
    }

    if (verbose)
      icepack::testing::print_errors(errors);

    CHECK(icepack::testing::is_decreasing(errors));
  }


  TEST_SUITE("derivative of membrane stress w.r.t. velocity")
  {
    const double T = 270.0;
    const SymmetricTensor<2, 2> eps{{1.0, -0.1, -0.1}};
    const SymmetricTensor<2, 2> deps{{-0.1, 0.5, -0.42}};

    const SymmetricTensor<2, 2> M0 = membrane_stress(T, eps);
    const SymmetricTensor<4, 2> dM = membrane_stress.du(T, eps);

    std::vector<double> errors(num_samples);
    for (unsigned int k = 0; k < num_samples; ++k)
    {
      const double delta = 1.0 / std::pow(2.0, k);
      const SymmetricTensor<2, 2> M = membrane_stress(T, eps + delta * deps);
      const SymmetricTensor<2, 2> M_approx = M0 + delta * dM * deps;
      errors[k] = (M - M_approx).norm();
    }

    if (verbose)
      icepack::testing::print_errors(errors);

    CHECK(icepack::testing::is_decreasing(errors));
  }

  return 0;
}

