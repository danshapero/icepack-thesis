
#include <icepack/physics/ice_stream.hpp>
#include <icepack/physics/constants.hpp>
#include "testing.hpp"
#include <fstream>

using dealii::Point;
namespace Functions = dealii::Functions;
using icepack::testing::Fn;
using namespace icepack::constants;

double average(const icepack::Field<2>& f)
{
  const icepack::Field<2> one =
    interpolate(f.discretization(), Functions::ConstantFunction<2>(1.0));
  return inner_product(f, one) / inner_product(one, one);
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

  const auto tria =
    icepack::testing::rectangular_mesh(L, W, num_levels, refined);
  const auto discretization = icepack::make_discretization(tria, p);
  const double tolerance = std::pow(icepack::testing::resolution(tria), p + 1);

  TEST_SUITE("computing surface elevation from bed + thickness")
  {
    const double H0 = 500.0;
    const double dH = 250.0;
    const Fn<2> Thickness([&](const Point<2>& x){return H0 - dH * x[0] / L;});

    const double B0 = -250.0;
    const double dB = -160.0;
    const Fn<2> Bed([&](const Point<2>& x){return B0 - dB * x[0] / L;});

    const Fn<2> Surface(
      [&](const Point<2>& x)
      {
        const double H = Thickness.value(x);
        const double B = Bed.value(x);
        return std::max(H + B, (1 - rho_ice / rho_water) * H);
      }
    );

    const icepack::Field<2> h = interpolate(*discretization, Thickness);
    const icepack::Field<2> b = interpolate(*discretization, Bed);
    const icepack::Field<2> s = interpolate(*discretization, Surface);
    CHECK_FIELDS(s, icepack::compute_surface(h, b), tolerance);
  }


  TEST_SUITE("coupled diagnostic/prognostic solves")
  {
    const double temp = 254.0;
    const Functions::ConstantFunction<2> Theta(temp);

    const double H0 = 500.0;
    const double dH = 100.0;
    const Fn<2> Thickness([&](const Point<2>& x){return H0 - dH * x[0] / L;});

    const double R = 1 - rho_ice / rho_water;
    const double height_above_flotation = 50.0;
    const Fn<2> Surface(
      [&](const Point<2>& x)
      {
        const double H = Thickness.value(x);
        return R * H + height_above_flotation;
      }
    );

    const double U0 = 100.0;
    const double dU = 25.0;
    const Fn<2> Ux([&](const Point<2>& x){return U0 + dU * x[0] / L;});

    const Fn<2> Beta([&](const Point<2>&){return 0.04;});

    const icepack::Field<2> h0 = interpolate(*discretization, Thickness);
    icepack::Field<2> h(h0);
    icepack::Field<2> s = interpolate(*discretization, Surface);
    const icepack::Field<2> b = s - h;

    const icepack::Field<2> theta = interpolate(*discretization, Theta);
    const icepack::Field<2> beta = interpolate(*discretization, Beta);

    const Functions::ZeroFunction<2> Uy;
    const icepack::VectorField<2> u0 = interpolate(*discretization, Ux, Uy);

    const std::set<dealii::types::boundary_id> dirichlet_ids{0, 2, 3};
    const icepack::IceStream ice_stream(dirichlet_ids);

    icepack::VectorField<2> u = ice_stream.solve(h, s, theta, beta, u0);

    icepack::Field<2> a(discretization);
    const double dt = icepack::compute_timestep(0.5, u);
    const double a_avg =
      average((ice_stream.solve(dt, h0, a, u, h0) - h0) / dt);
    a = interpolate(*discretization, Functions::ConstantFunction<2>(a_avg));

    if (verbose)
      std::cout << "Mean accumulation rate: " << a_avg << "\n";

    const size_t num_timesteps = 12;
    for (size_t n = 0; n < num_timesteps; ++n)
    {
      h = ice_stream.solve(dt, h, a, u, h0);
      s = icepack::compute_surface(h, b);
      u = ice_stream.solve(h, s, theta, beta, u);
    }

    // TODO: Find an exactly solvable case. All this does is check that nothing
    // failed so catastrophically that the solvers hang or we get an exception.
    if (verbose)
    {
      icepack::Field<2> dh = h - h0;
      std::cout << "Thickness change: " << max(dh) << "\n";

      std::ofstream h_file("h.ucd");
      h.write_ucd("h", h_file);
    }
  }

  return 0;
}

