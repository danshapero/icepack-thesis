
#include <icepack/physics/ice_stream.hpp>
#include <icepack/physics/constants.hpp>
#include "testing.hpp"

using dealii::Point;
using dealii::Tensor;
using dealii::SymmetricTensor;
using icepack::testing::Fn;
using icepack::testing::TensorFn;
using namespace icepack::constants;

int main(int argc, char ** argv)
{
  const auto args = icepack::testing::get_cmdline_args(argc, argv);
  const bool verbose = args.count("--verbose");
  const bool refined = args.count("--refined");
  const bool quadratic = args.count("--quadratic");

  const size_t num_levels = 5 - quadratic;
  const size_t p = 1 + quadratic;

  icepack::MembraneStress membrane_stress;
  const double n = membrane_stress.n;

  icepack::BasalStress basal_stress;
  const double m = basal_stress.m;

  const double L = 2000.0;
  const double W = 500.0;

  const double temp = 254.0;
  const Fn<2> Theta([&](const Point<2>&){ return temp; });

  const double h0 = 500.0;
  const double dh = 100.0;
  const Fn<2> Thickness([&](const Point<2>& x){ return h0 - dh * x[0] / L; });

  const double height_above_flotation = 50.0;
  const double s0 = (1 - rho_ice / rho_water) * h0 + height_above_flotation;
  const double ds = (1 - rho_ice / rho_water) * dh + height_above_flotation;
  const Fn<2> Surface([&](const Point<2>& x){ return s0 - ds * x[0] / L; });

  const double rho = rho_ice * (1 - rho_ice / rho_water);
  const double A =
    std::pow(rho * gravity * h0 / 4, 3) * icepack::rate_factor(temp);
  const double u0 = 100.0;
  const double gamma = dh / h0;
  const double alpha = gamma / A;

  const TensorFn<2> Velocity(
    [&](const Point<2>& x)
    {
      const double q = std::pow(1 - gamma * x[0] / L, 4);
      return Tensor<1, 2>{{u0 + L / (4 * alpha) * (1 - q), 0.0}};
    }
  );

  const TensorFn<2> InitialVelocity(
    [&](const Point<2>& x)
    {
      const Tensor<1, 2> v = Velocity.value(x);
      const double px = x[0] / L;
      const double py = x[1] / W;
      const double ax = px * (1 - px);
      const double ay = py * (1 - py);
      return v + ax * ay * 500.0 * Tensor<1, 2>{{1.0, 0.5 - py}};
    }
  );

  const TensorFn<2> MembraneStressDivergence(
    [&](const Point<2>& x)
    {
      const double B = membrane_stress.B(temp);
      const double dh_dx = -dh / L;
      const double h = Thickness.value(x);
      const double q = (1 - gamma * x[0] / L) * dh_dx - gamma / L * h;
      return Tensor<1, 2> {{2 * B * std::pow(gamma / alpha, 1/n) * q, 0.0}};
    }
  );

  const TensorFn<2> DrivingStress(
    [&](const Point<2>& x)
    {
      const Tensor<1, 2> grad_s{{-ds / L, 0.0}};
      return -rho_ice * gravity * Thickness.value(x) * grad_s;
    }
  );

  const Fn<2> Beta(
    [&](const Point<2>& x)
    {
      const Tensor<1, 2> tau_m = MembraneStressDivergence.value(x);
      const Tensor<1, 2> tau_d = DrivingStress.value(x);
      const Tensor<1, 2> u = Velocity.value(x);
      return (tau_m[0] + tau_d[0]) / std::pow(u[0], 1/m);
    }
  );

  const dealii::Triangulation<2> tria =
    icepack::testing::rectangular_mesh(L, W, num_levels, refined);
  const icepack::Discretization<2> discretization(tria, p);
  const double tolerance = std::pow(icepack::testing::resolution(tria), p + 1);

  const icepack::Field<2> h = interpolate(discretization, Thickness);
  const icepack::Field<2> s = interpolate(discretization, Surface);
  const icepack::Field<2> theta = interpolate(discretization, Theta);
  const icepack::Field<2> beta = interpolate(discretization, Beta);
  const icepack::VectorField<2> u = interpolate(discretization, Velocity);
  const icepack::VectorField<2> du =
    interpolate(discretization, InitialVelocity);

  const std::set<dealii::types::boundary_id> dirichlet_ids{0, 2, 3};
  const icepack::IceStream ice_stream(dirichlet_ids);

  size_t iterations = 0;
  const auto callback =
    [&](const double action, const icepack::VectorField<2>&)
    {
      if (verbose)
        std::cout << " " << iterations << ": " << action << "\n";
      iterations += 1;
    };

  const icepack::IceStream::SolveOptions options{callback};
  const icepack::VectorField<2> v =
    ice_stream.solve(h, s, theta, beta, du, options);

  CHECK_FIELDS(u, v, tolerance);
  CHECK(iterations != 0);

  return 0;
}
