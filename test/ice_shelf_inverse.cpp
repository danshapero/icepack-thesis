
#include <icepack/physics/ice_shelf.hpp>
#include <icepack/physics/constants.hpp>
#include "testing.hpp"

using dealii::Point;
using dealii::Tensor;
using icepack::testing::Fn;
using icepack::testing::TensorFn;
using namespace icepack::constants;

int main(int argc, char ** argv)
{
  const auto args = icepack::testing::get_cmdline_args(argc, argv);
  //const bool verbose = args.count("--verbose");
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
  const Fn<2> Theta([&](const Point<2>&){return temp;});

  const double rho = rho_ice * (1 - rho_ice / rho_water);
  const icepack::MembraneStress membrane_stress;
  const double n = membrane_stress.n;
  const double A = icepack::rate_factor(temp);
  const double u0 = 100.0;
  const TensorFn<2> Velocity(
    [&](const Point<2>& x)
    {
      const double q = 1 - pow(1 - dh * x[0] / (L * h0), 4);
      const double f = rho * gravity * h0;
      return Tensor<1, 2>{{u0 + std::pow(f / 4, n) * A * q * L / dh / 4, 0}};
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
     return alpha * dtheta;
    }
  );


  const dealii::Triangulation<2> tria =
    icepack::testing::rectangular_mesh(L, W, num_levels, refined);
  const auto discretization = icepack::make_discretization(tria, p);

  const icepack::Field<2> h = interpolate(*discretization, Thickness);
  const icepack::Field<2> theta = interpolate(*discretization, Theta);
  const icepack::VectorField<2> u = interpolate(*discretization, Velocity);

  const icepack::Field<2> phi = interpolate(*discretization, DTheta);
  const icepack::VectorField<2> v = interpolate(*discretization, DVelocity);

  const icepack::Viscosity viscosity(membrane_stress);

  const icepack::DualField<2> dF_dtheta =
    viscosity.mixed_derivative(h, theta, u, v);

  std::cout << inner_product(dF_dtheta, phi) << "\n";

  return 0;
}

