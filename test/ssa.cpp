
#include <icepack/physics/constants.hpp>
#include <icepack/physics/ssa.hpp>
#include "testing.hpp"

using dealii::Point;
using dealii::Tensor;
using dealii::SymmetricTensor;

void print_diffs(const std::vector<double>& diffs)
{
  for (const auto diff: diffs)
    std::cout << diff << " ";
  std::cout << "\n";
}


int main(int argc, char ** argv)
{
  const auto args = icepack::testing::get_cmdline_args(argc, argv);
  const bool verbose = args.count("--verbose");
  const bool refined = args.count("--refined");
  const bool quadratic = args.count("--quadratic");
  const size_t num_samples = 8;

  icepack::ViscousRheology rheology;

  TEST_SUITE("derivative of rheology")
  {
    const auto test = [&](const double T0, const double dT)
    {
      const double B0 = rheology(T0);
      const double dB = rheology.dtheta(T0);

      std::vector<double> diffs(num_samples);
      for (unsigned int k = 0; k < num_samples; ++k)
      {
        const double delta = 1.0 / std::pow(2.0, k);
        const double B = rheology(T0 + delta * dT);
        const double B_approx = B0 + delta * dB * dT;
        diffs[k] = std::abs(B - B_approx);
      }

      if (verbose)
        print_diffs(diffs);

      CHECK(icepack::testing::is_decreasing(diffs));
    };

    // Check that the derivative of the rheology is implemented right both above
    // and below the transition temp of 263.15K.
    test(253.0, 5.0);
    test(270.0, -5.0);
  }


  icepack::MembraneStress membrane_stress(rheology);

  TEST_SUITE("derivative of membrane stress w.r.t. temperature")
  {
    const double T0 = 253.0;
    const double dT = 5.0;
    const SymmetricTensor<2, 2> eps{{0.5, -0.8, 2.3}};

    const SymmetricTensor<2, 2> M0 = membrane_stress(T0, eps);
    const SymmetricTensor<2, 2> dM = membrane_stress.dtheta(T0, eps);

    std::vector<double> diffs(num_samples);
    for (unsigned int k = 0; k < num_samples; ++k)
    {
      const double delta = 1.0 / std::pow(2.0, k);
      const SymmetricTensor<2, 2> M = membrane_stress(T0 + delta * dT, eps);
      const SymmetricTensor<2, 2> M_approx = M0 + delta * dM * dT;
      diffs[k] = (M - M_approx).norm();
    }

    if (verbose)
      print_diffs(diffs);

    CHECK(icepack::testing::is_decreasing(diffs));
  }


  TEST_SUITE("derivative of membrane stress w.r.t. velocity")
  {
    const double T = 270.0;
    const SymmetricTensor<2, 2> eps{{1.0, -0.1, -0.1}};
    const SymmetricTensor<2, 2> deps{{-0.1, 0.5, -0.42}};

    const SymmetricTensor<2, 2> M0 = membrane_stress(T, eps);
    const SymmetricTensor<4, 2> dM = membrane_stress.du(T, eps);

    std::vector<double> diffs(num_samples);
    for (unsigned int k = 0; k < num_samples; ++k)
    {
      const double delta = 1.0 / std::pow(2.0, k);
      const SymmetricTensor<2, 2> M = membrane_stress(T, eps + delta * deps);
      const SymmetricTensor<2, 2> M_approx = M0 + delta * dM * deps;
      diffs[k] = (M - M_approx).norm();
    }

    if (verbose)
      print_diffs(diffs);

    CHECK(icepack::testing::is_decreasing(diffs));
  }


  /**
   * Set up a whole mess of exact data to compare against model output.
   * See Greve and Blatter, "Dynamics of Ice Sheets and Glaciers", chapter 6
   * for the exact solution of an ice shelf ramp.
   */
  using namespace icepack::constants;
  const double rho = rho_ice * (1 - rho_ice / rho_water);

  const double length = 2000.0;
  const double width = 500.0;
  const double u0 = 100.0;
  const double h0 = 500.0;
  const double dh = 100.0;
  const double temp = 254.0;
  const double A = std::pow(rho * gravity / 4, 3) * icepack::rate_factor(temp);
  const double area = length * width;

  // If we're testing with biquadratic finite elements, use a coarser mesh.
  const size_t num_levels = 5 - quadratic;

  // The degree of the finite element basis functions we're using. The accuracy
  // we expect for all our methods depends on the order of the basis functions.
  const size_t p = 1 + quadratic;

  const auto tria =
    icepack::testing::rectangular_mesh(length, width, num_levels, refined);
  const icepack::Discretization<2> discretization(tria, p);
  const double tolerance = std::pow(icepack::testing::resolution(tria), p + 1);

  // Use a constant temperature, and linearly decreasing thickness.
  using icepack::testing::Fn;
  const auto Theta =
    Fn<2>([&](const Point<2>&){return temp;});
  const auto Thickness =
    Fn<2>([&](const Point<2>& x){return h0 - dh * x[0] / length;});

  // The solution of the shallow shelf equations with this temperature and
  // thickness can be computed analytically; we'll use this to compare our
  // numerical solutions against.
  using icepack::testing::TensorFn;
  const auto Velocity =
    TensorFn<2>([&](const Point<2>& x)
                {
                  const double q = 1 - pow(1 - dh * x[0] / (length * h0), 4);
                  Tensor<1, 2> v{{u0 + A * q * length * pow(h0, 4) / dh / 4, 0}};
                  return v;
                });

  /**
   * Interpolate all the exact data to the finite element basis.
   */
  const icepack::Field<2> theta = interpolate(discretization, Theta);
  const icepack::Field<2> h = interpolate(discretization, Thickness);
  const icepack::VectorField<2> u = interpolate(discretization, Velocity);

  const icepack::Viscosity viscosity(rheology);

  const double n = rheology.n;
  const double B = rheology(temp);
  const double h_integral =
    (std::pow(h0, n + 3) - std::pow(h0 - dh, n + 3))/((n + 3) * dh);
  const double power =
    area * std::pow(rho * gravity, n + 1) / std::pow(4*B, n) * h_integral;


  /**
   * Check the values of the action functional against analytically computed
   * values using the thickness and velocity for the ice shelf ramp.
   */

  TEST_SUITE("viscous action functional")
  {
    const double exact_viscous_power = n/(n + 1) * power / 2;
    const double viscous_power = viscosity.action(h, theta, u);

    CHECK_REAL(exact_viscous_power, viscous_power, tolerance * area);
  }


  const icepack::Gravity gravity;

  TEST_SUITE("gravitational driving stress action functional")
  {
    const double exact_gravity_power = -0.5 * power;
    const double gravity_power = gravity.action(h, u);

    CHECK_REAL(exact_gravity_power, gravity_power, tolerance * area);
  }


  /**
   * Make a perturbation to the velocity field for the following tests.
   */
  const auto DVelocity =
    TensorFn<2>([&](const Point<2>& x)
                {
                  Tensor<1, 2> v = Velocity.value(x);
                  const double px = x[0]/length;
                  const double py = x[1] / width;
                  const double ax = px * (1 - px);
                  const double ay = py * (1 - py);
                  v[0] += ax * ay * 500.0;
                  v[1] += ax * ay * (0.5 - py) * 500.0;
                  return v;
                });

  const icepack::VectorField<2> du = interpolate(discretization, DVelocity);


  /**
   * Check the derivatives of the action functionals.
   */
  TEST_SUITE("derivative of viscous stress")
  {
    const double P = viscosity.action(h, theta, u);
    const icepack::DualVectorField<2> dP = viscosity.derivative(h, theta, u);
    const dealii::SparseMatrix<double> d2P = viscosity.hessian(h, theta, u);

    const double linear_term = inner_product(dP, du);
    CHECK_REAL(linear_term, viscosity.derivative(h, theta, u, du), 1.0e-6);

    const double quadratic_term = d2P.matrix_norm_square(du.coefficients());

    const size_t num_trials = 12;
    std::vector<double> errors(num_trials);
    for (size_t k = 0; k < num_trials; ++k)
    {
      const double delta = 1.0 / pow(2.0, k);

      const icepack::VectorField<2> v = u + delta * du;
      const double P_exact = viscosity.action(h, theta, v);

      const double P_approx =
          P + delta * linear_term + 0.5 * std::pow(delta, 2) * quadratic_term;
      const double error = std::abs(P_exact - P_approx);

      errors[k] = error / (std::pow(delta, 2) * std::abs(P));
    }

    if (verbose)
      print_diffs(errors);

    CHECK(icepack::testing::is_decreasing(errors));
  }


  TEST_SUITE("derivative of gravitational stress")
  {
    const double P = gravity.action(h, u);
    const icepack::DualVectorField<2> dP = gravity.derivative(h);

    const double linear_term = inner_product(dP, du);
    CHECK_REAL(linear_term, gravity.derivative(h, du), 1.0e-6);

    const size_t num_trials = 12;
    std::vector<double> errors(num_trials);
    for (size_t k = 0; k < num_trials; ++k)
    {
      const double delta = 1.0 / pow(2.0, k);

      const icepack::VectorField<2> v = u + delta * du;
      const double P_exact = gravity.action(h, v);

      const double P_approx = P + delta * linear_term;
      const double error = std::abs(P_exact - P_approx);

      errors[k] = error / (delta * std::abs(P));
    }

    if (verbose)
      print_diffs(errors);

    // The gravitational stress action is linear in the velocity, so we should
    // check that the error in the linear approximation is always small, not
    // that it's decreasing.
    CHECK(std::all_of(errors.begin(), errors.end(),
                      [=](const auto err){return err < tolerance;}));
  }


  TEST_SUITE("solving the diagnostic equations")
  {
    const std::set<dealii::types::boundary_id> dirichlet_boundary_ids{0, 2, 3};
    const icepack::IceShelf ice_shelf(dirichlet_boundary_ids);

    size_t iterations = 0;
    const auto callback =
      [&](const double action, const icepack::VectorField<2>&)
      {
        if (verbose)
          std::cout << "  " << iterations << ": " << action << "\n";
        iterations += 1;
      };

    const icepack::IceShelf::SolveOptions options{callback};
    const icepack::VectorField<2> v = ice_shelf.solve(h, theta, du, options);

    CHECK_FIELDS(u, v, tolerance);
    CHECK(iterations != 0);
  }

  return 0;
}
