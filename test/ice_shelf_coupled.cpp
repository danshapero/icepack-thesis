
#include <icepack/physics/ice_shelf.hpp>
#include <icepack/physics/constants.hpp>
#include "testing.hpp"
#include <fstream>

using dealii::Point;
using dealii::Tensor;
using dealii::SymmetricTensor;

int main(int argc, char ** argv)
{
  const auto args = icepack::testing::get_cmdline_args(argc, argv);
  const bool verbose = args.count("--verbose");
  const bool refined = args.count("--refined");
  const bool quadratic = args.count("--quadratic");

  using namespace icepack::constants;
  const double rho = rho_ice * (1 - rho_ice / rho_water);

  const double L = 2000.0;
  const double W = 500.0;
  const double U0 = 100.0;
  const double H0 = 500.0;
  const double temp = 254.0;

  icepack::MembraneStress membrane_stress;
  const double n = membrane_stress.n;
  const double B = membrane_stress.B(temp);

  const size_t num_levels = 5 - quadratic;
  const size_t p = 1 + quadratic;

  const dealii::Triangulation<2> tria =
    icepack::testing::rectangular_mesh(L, W, num_levels, refined);
  const auto discretization = icepack::make_discretization(tria, p);
  const double tolerance = std::pow(icepack::testing::resolution(tria), p + 1);


  /**
   * These velocity and thickness fields are both in perfect mass balance with
   * no surface accumulation.
   */
  const icepack::testing::TensorFn<2> Velocity(
    [&](const Point<2>& x)
    {
      const double q = (n + 1) * std::pow(rho * gravity * H0 * U0 / (4 * B), n);
      const double v = std::pow(std::pow(U0, n + 1) + q * x[0], 1 / (n + 1));
      return Tensor<1, 2>{{v, 0}};
    }
  );


  const icepack::testing::Fn<2> Thickness(
    [&](const Point<2>& x)
    {
      const double q = (n + 1) * std::pow(rho * gravity * H0 * U0 / (4 * B), n);
      const double v = std::pow(std::pow(U0, n + 1) + q * x[0], 1 / (n + 1));
      return H0 * U0 / v;
    }
  );


  /**
   * Interpolate the exact fields to the finite element representation and
   * check that they're roughly in steady state.
   */
  const icepack::Field<2> h0 = interpolate(*discretization, Thickness);
  const icepack::VectorField<2> u0 = interpolate(*discretization, Velocity);

  icepack::Field<2> h(h0);
  icepack::VectorField<2> u(u0);

  const icepack::Field<2> a =
    interpolate(*discretization, dealii::ZeroFunction<2>());
  const icepack::Field<2> theta =
    interpolate(*discretization, dealii::ConstantFunction<2>(temp));

  const icepack::Viscosity viscosity(membrane_stress);
  const std::set<dealii::types::boundary_id> dirichlet_boundary_ids{0, 2, 3};
  const icepack::IceShelf ice_shelf(dirichlet_boundary_ids, viscosity);

  const double dt = icepack::compute_timestep(1.0, u);

  const size_t num_timesteps = 10;
  for (size_t n = 0; n < num_timesteps; ++n)
  {
    if (verbose)
      std::cout << n << " ";

    h = ice_shelf.solve(dt, h, a, u, h0);
    u = ice_shelf.solve(h, theta, u0);

    if (verbose)
    {
      std::ofstream h_file("h" + std::to_string(n) + ".ucd");
      h.write_ucd("h", h_file);

      std::ofstream u_file("u" + std::to_string(n) + ".ucd");
      u.write_ucd("u", u_file);
    }

    CHECK_FIELDS(h, h0, tolerance);
    CHECK_FIELDS(u, u0, tolerance);
  }

  if (verbose)
    std::cout << "\n";

  return 0;
}

