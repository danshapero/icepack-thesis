
#include <icepack/physics/ice_shelf.hpp>
#include <icepack/physics/constants.hpp>
#include <icepack/inverse/error_functionals.hpp>
#include "testing.hpp"

using dealii::Point;
using dealii::Tensor;
using icepack::testing::Fn;
using icepack::testing::TensorFn;
using namespace icepack::constants;


/**
 * Use a simpler parameterization of the ice fluidity factor than the usual
 * function of temperature -- when solving inverse problems, we usually infer
 * `A` or the rheology coefficient `B` directly and back out the temperature
 * after the fact.
 */
double AA(const double theta)
{
    return theta;
}

double dAA(const double)
{
    return 1.0;
}


int main(int argc, char ** argv)
{
  const auto args = icepack::testing::get_cmdline_args(argc, argv);
  const bool verbose = args.count("--verbose");
  const bool refined = args.count("--refined");
  const bool quadratic = args.count("--quadratic");

  const size_t num_levels = 5 - quadratic;
  const size_t p = 1 + quadratic;

  const double L = 2000.0;
  const double W = 500.0;

  const double h0 = 500.0;
  const double dh = 100.0;
  const Fn<2> Thickness([&](const Point<2>& x){return h0 - dh * x[0] / L;});

  const double temp = 254.0;
  const double A = icepack::rate_factor(temp);
  const Fn<2> Theta([&](const Point<2>&){return A;});

  const double rho = rho_ice * (1 - rho_ice / rho_water);
  const icepack::MembraneStress membrane_stress(3.0, AA, dAA);
  const double n = membrane_stress.n;
  const double u0 = 100.0;
  const TensorFn<2> Velocity(
    [&](const Point<2>& x)
    {
      const double q = 1 - pow(1 - dh * x[0] / (L * h0), 4);
      const double f = rho * gravity * h0;
      return Tensor<1, 2>{{u0 + std::pow(f / 4, n) * A * q * L * h0 / dh / 4, 0}};
    }
  );

  const TensorFn<2> DVelocity(
    [&](const Point<2>& x)
    {
      const double px = x[0] / L;
      const double py = x[1] / W;
      const double alpha = 16 * px * (1 - px) * py * (1 - py);
      return alpha * u0 * Tensor<1, 2>{{0, (2 * py - 1)}};
    }
  );

  const Fn<2> DTheta(
    [&](const Point<2>& x)
    {
     const double px = x[0] / L;
     const double py = x[1] / W;
     const double alpha = 16 * px * (1 - px) * py * (1 - py);
     const double dtheta = 5.0;
     return icepack::rate_factor(temp + alpha * dtheta) - A;
    }
  );


  const dealii::Triangulation<2> tria =
    icepack::testing::rectangular_mesh(L, W, num_levels, refined);
  const auto discretization = icepack::make_discretization(tria, p);

  const icepack::Field<2> h = interpolate(*discretization, Thickness);
  const icepack::Viscosity viscosity(membrane_stress);

  TEST_SUITE("mixed derivatives of the viscous action functional")
  {
    const icepack::Field<2> theta = interpolate(*discretization, Theta);
    const icepack::VectorField<2> u = interpolate(*discretization, Velocity);

    const icepack::Field<2> phi = interpolate(*discretization, DTheta);
    const icepack::VectorField<2> v = interpolate(*discretization, DVelocity);

    const icepack::DualField<2> dF_dtheta =
      viscosity.mixed_derivative(h, theta, u, v);
    const double dF = inner_product(dF_dtheta, phi);

    const size_t num_samples = 16;
    std::vector<double> errors(num_samples);
    for (size_t k = 0; k < num_samples; ++k)
    {
      const double delta = 1.0 / (1 << k);
      const double delta_F =
        (inner_product(viscosity.derivative(h, theta + delta * phi, u), v) -
         inner_product(viscosity.derivative(h, theta - delta * phi, u), v))
        / (2*delta);
      errors[k] = std::abs(delta_F - dF);
    }

    if (verbose)
      icepack::testing::print_errors(errors);

    CHECK(icepack::testing::is_decreasing(errors));
  }


  TEST_SUITE("derivative of the velocity w.r.t. temperature")
  {
    // Guess/perturbation of the fluidity parameter
    const icepack::Field<2> theta = interpolate(*discretization, Theta);
    const icepack::Field<2> dtheta = interpolate(*discretization, DTheta);

    // Guess for the velocity
    const icepack::VectorField<2> u = interpolate(*discretization, Velocity);

    // Compute the true velocity by solving the diagnostic equations with the
    // given fluidity
    const std::set<dealii::types::boundary_id> dirichlet_boundary_ids{0, 2, 3};
    const icepack::IceShelf ice_shelf(dirichlet_boundary_ids, viscosity);
    const icepack::VectorField<2> uo = ice_shelf.solve(h, theta + dtheta, u);

    // Create a field for the standard deviation
    const icepack::Field<2> sigma =
      interpolate(*discretization, Fn<2>([](const Point<2>&){return 1.0;}));

    // Compute the misfit between the velocities `u` and `uo`
    const icepack::MeanSquareError<2> mean_square_error{};
    const double misfit = mean_square_error.action(u, uo, sigma);
    if (verbose)
      std::cout << "Average misfit between true and guessed velocities: "
                << misfit / (L * W) << "\n";

    const dealii::ConstraintMatrix& constraints =
      discretization->vector().constraints();

    // Compute the derivative of the misfit
    const icepack::DualVectorField<2> dE =
      -mean_square_error.derivative(u, uo, sigma, constraints);

    // Solve for the adjoint state variable
    const dealii::SparseMatrix<double> A =
      viscosity.hessian(h, theta, u, constraints);

    dealii::SparseDirectUMFPACK G;
    G.initialize(A);
    icepack::VectorField<2> lambda(discretization);
    G.vmult(lambda.coefficients(), dE.coefficients());
    constraints.distribute(lambda.coefficients());

    // Compute the derivative of the misfit functional with respect to `theta`
    // using the adjoint state variable
    const icepack::DualField<2> dF =
      viscosity.mixed_derivative(h, theta, u, lambda);

    std::cout << "Directional derivative: " << inner_product(dF, dtheta) << "\n";

    std::cout << "Max values of theta: " << max(theta) << ", "
              << icepack::max(icepack::Field<2>(theta + dtheta)) << "\n";

    const size_t num_samples = 12;
    std::vector<double> errors(num_samples);
    for (size_t k = 0; k < num_samples; ++k)
    {
      const double delta = 1.0 / (1 << k);
      const icepack::Field<2> phi = theta + delta * dtheta;
      const icepack::VectorField<2> v = ice_shelf.solve(h, phi, u);
      std::cout << mean_square_error.action(v, uo, sigma) - misfit << ", "
                << delta * inner_product(dF, dtheta) << "\n";
      const double delta_E = mean_square_error.action(v, uo, sigma) - misfit;
      errors[k] = std::abs(delta_E - delta * inner_product(dF, dtheta));
    }
    std::cout << "\n";

    /*
    if (verbose)
      icepack::testing::print_errors(errors);

    CHECK(icepack::testing::is_decreasing(errors));
    */
  }

  return 0;
}

