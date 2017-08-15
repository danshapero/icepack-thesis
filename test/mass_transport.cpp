
#include <icepack/physics/constants.hpp>
#include <icepack/physics/mass_transport.hpp>
#include "testing.hpp"
#include <deal.II/base/function.h>

using dealii::Point;
using dealii::Tensor;
using icepack::testing::Fn;
using icepack::testing::TensorFn;


struct DifferentiableFn : public Fn<2>
{
  template <typename Q, typename dQ>
  DifferentiableFn(Q&& q, dQ&& dq) :
    Fn<2>(q), derivative(std::forward<dQ>(dq))
  {}

  Fn<2> derivative;
};


TensorFn<2> tensor_fn_from_coords(const Fn<2>& U1, const Fn<2>& U2)
{
  const auto u =
    [&](const Point<2>& x)
    {
      return Tensor<1, 2>{{U1.value(x), U2.value(x)}};
    };

  return TensorFn<2>(u);
}


int main(int argc, char ** argv)
{
  const auto args = icepack::testing::get_cmdline_args(argc, argv);
  const bool verbose = args.count("--verbose");
  const bool refined = args.count("--refined");
  const bool quadratic = args.count("--quadratic");

  const double length = 2000.0;
  const double width = 500.0;
  const double u0 = 100.0;
  const double du = 100.0;
  const double h_in = 500.0;
  const double dh = 100.0;

  // If we're testing with biquadratic finite elements, use a coarser mesh.
  const size_t num_levels = 5 - quadratic;

  // The degree of the finite element basis functions we're using. The accuracy
  // we expect for all our methods depends on the order of the basis functions.
  const size_t p = 1 + quadratic;

  const dealii::Triangulation<2> tria =
    icepack::testing::rectangular_mesh(length, width, num_levels, refined);
  const auto dptr = icepack::make_discretization(tria, p);
  const icepack::Discretization<2>& discretization = *dptr;
  const double tolerance = std::pow(icepack::testing::resolution(tria), p);

  const icepack::MassTransportImplicit mass_transport;


  TEST_SUITE("mass transport, implicit discretization")
  {
    const auto& test_flux =
      [&](const DifferentiableFn& Thickness, const DifferentiableFn& Velocity)
      {
        const Fn<2> Flux(
          [&](const Point<2>& x)
          {
            const double H = Thickness.value(x);
            const double dH_dx = Thickness.derivative.value(x);
            const double U = Velocity.value(x);
            const double dU_dx = Velocity.derivative.value(x);
            return H * dU_dx + U * dH_dx;
          }
        );

        const icepack::Field<2> flux_exact = interpolate(discretization, Flux);

        const Fn<2> V([&](const Point<2>&){ return 0.0; });
        const icepack::VectorField<2> u =
          interpolate(discretization, tensor_fn_from_coords(Velocity, V));

        const dealii::SparseMatrix<double> F = mass_transport.flux_matrix(u);
        const icepack::Field<2> h = interpolate(discretization, Thickness);
        icepack::DualField<2> f(discretization.shared_from_this());
        F.vmult(f.coefficients(), h.coefficients());
        const icepack::Field<2> flux = transpose(f);

        CHECK_FIELDS(flux, flux_exact, tolerance);
      };


    TEST_SUITE("constant velocity, linear thickness")
    {
      const DifferentiableFn Thickness(
        [&](const Point<2>& x){ return h_in - dh * x[0] / length; },
        [&](const Point<2>&){ return -dh / length; }
      );

      const DifferentiableFn Velocity(
        [&](const Point<2>&){ return u0; },
        [&](const Point<2>&){ return 0.0; }
      );

      test_flux(Thickness, Velocity);
    }


    TEST_SUITE("linear velocity, linear thickness")
    {
      const DifferentiableFn Thickness(
        [&](const Point<2>& x){ return h_in - dh * x[0] / length; },
        [&](const Point<2>&){ return -dh / length; }
      );

      const DifferentiableFn Velocity(
        [&](const Point<2>& x){ return u0 + du * x[0] / length; },
        [&](const Point<2>&){ return du / length; }
      );

      test_flux(Thickness, Velocity);
    }
  }


  TEST_SUITE("computing a timestep")
  {
    const TensorFn<2> Velocity(
      [&](const Point<2>& x)
      {
        return Tensor<1, 2>{{u0 + du * x[0] / length, 0}};
      }
    );
    const icepack::VectorField<2> u = interpolate(discretization, Velocity);

    const double courant_number = 0.5;
    const double dt = icepack::compute_timestep(courant_number, u);

    const double u_max = max(u);
    const auto& tria = discretization.triangulation();
    for (const auto& cell: tria.active_cell_iterators())
      CHECK(u_max * dt / cell->diameter() <= courant_number);
  }


  const auto test_solver =
    [&, verbose](
      const auto& H0,
      const auto& U,
      const auto& A,
      const auto& H_inflow,
      const auto& H
    )
    {
      const icepack::Field<2> h0 = interpolate(discretization, H0);
      const icepack::VectorField<2> u = interpolate(discretization, U);

      const double courant_number = 0.5;
      const double dt = icepack::compute_timestep(courant_number, u);

      const double T = length / u0;
      const size_t num_timesteps = std::ceil(T / dt);

      if (verbose)
        std::cout << "Timestep:            " << dt << "\n"
                  << "Residence time:      " << T << "\n"
                  << "Number of timesteps: " << num_timesteps << "\n";

      const std::set<dealii::types::boundary_id> bcs{0};

      icepack::Field<2> h(h0);
      for (size_t k = 0; k < num_timesteps; ++k)
      {
        const double t = k * dt;
        const icepack::Field<2> a = interpolate(discretization, A(t));
        const icepack::Field<2> h_inflow =
          interpolate(discretization, H_inflow(t));

        h = mass_transport.solve(dt, h, a, u, bcs, h_inflow);
      }

      if (verbose)
        std::cout << "Final max thickness: " << max(h) << "\n";

      const icepack::Field<2> h_exact =
        interpolate(discretization, H(num_timesteps * dt));

      CHECK_FIELDS(h, h_exact, std::max(dt / T, tolerance));

      return h;
    };


  TEST_SUITE("solving the advection equation")
  {
    TEST_SUITE("linear thickness, constant velocity")
    {
      const Fn<2> H0(
        [&](const Point<2>& x)
        {
          return h_in - dh * x[0] / length;
        }
      );

      const auto H_inflow =
        [&](const double) -> Fn<2>
        {
          return H0;
        };

      const TensorFn<2> U(
        [&](const Point<2>&)
        {
          return Tensor<1, 2>{{u0, 0.0}};
        }
      );

      const auto A = [&](const double){ return dealii::ZeroFunction<2>(); };

      const auto H =
        [&](const double t) -> Fn<2>
        {
          const auto H =
            [&, t](const Point<2>& x)
            {
              if (x[0] < u0 * t)
                return h_in;
              return h_in - dh * (x[0] - u0 * t) / length;
            };

          return Fn<2>(H);
        };

      const icepack::Field<2> h = test_solver(H0, U, A, H_inflow, H);

      CHECK(max(h) <= h_in * (1 + tolerance));
    }


    TEST_SUITE("changing inflow thickness")
    {
      const Fn<2> H0(
        [&](const Point<2>&)
        {
          return h_in;
        }
      );

      const double T = length / u0;
      const auto H_inflow =
        [&](const double t) -> Fn<2>
        {
          const auto H =
            [&](const Point<2>&)
            {
              if (t < T)
                return h_in + dh * t / T;
              return h_in + dh;
            };

          return Fn<2>(H);
        };

      const TensorFn<2> U(
        [&](const Point<2>&)
        {
          return Tensor<1, 2>{{u0, 0.0}};
        }
      );

      const auto A = [&](const double){ return dealii::ZeroFunction<2>(); };

      const auto H =
        [&](const double t) -> Fn<2>
        {
          const auto H =
            [&, t](const Point<2>& x)
            {
              if (x[0] < u0 * t)
                return H_inflow(t - x[0] / u0).value(Point<2>{0.0, x[1]});
              return h_in;
            };

          return Fn<2>(H);
        };

      test_solver(H0, U, A, H_inflow, H);
    }


    // Note: you can solve this by the method of characteristics.
    TEST_SUITE("linear thickness, linear velocity")
    {
      const Fn<2> H0(
        [=](const Point<2>& x)
        {
          return h_in - dh * x[0] / length;
        }
      );

      const auto H_inflow =
        [h_in](const double){ return dealii::ConstantFunction<2>(h_in); };

      const TensorFn<2> U(
        [=](const Point<2>& x)
        {
          return Tensor<1, 2>{{u0 + du * x[0] / length, 0}};
        }
      );

      const auto A = [](const double){ return dealii::ZeroFunction<2>(); };

      // Given some point `x` inside the domain, `T(x)` is the length of time
      // that this point has spent to the right of the origin.
      const auto T =
        [=](const double x)
        {
          return length / du * std::log(1 + du / u0 * x / length);
        };

      const auto Xi =
        [=](const Point<2>& x, const double t)
        {
          const double q = std::exp(-du * t / length);
          Point<2> xi{x[0] * q - u0 / du * length * (1 - q), x[1]};
          xi[0] = std::max(xi[0], 0.0);
          return xi;
        };

      const auto H =
        [&](const double t) -> Fn<2>
        {
          const auto H =
            [&, t](const Point<2>& x)
            {
              const double T0 = T(x[0]);
              if (t < T0)
              {
                const Point<2> xi = Xi(x, t);
                return H0.value(xi) * std::exp(-du / length * t);
              }

              return h_in * std::exp(-du / length * T0);
            };

          return Fn<2>(H);
        };

      test_solver(H0, U, A, H_inflow, H);
    }


    TEST_SUITE("linear thickness, constant velocity, source terms")
    {
      const Fn<2> H0(
        [&](const Point<2>& x)
        {
          return h_in - dh * x[0] / length;
        }
      );

      const auto H_inflow =
        [&](const double) -> Fn<2>
        {
          return H0;
        };

      const TensorFn<2> U(
        [&](const Point<2>&)
        {
          return Tensor<1, 2>{{u0, 0.0}};
        }
      );

      const double a_dot = 0.1 * h_in * u0 / length;
      const auto A =
        [=](const double)
        {
          return dealii::ConstantFunction<2>(a_dot);
        };


      const auto H =
        [&](const double t) -> Fn<2>
        {
          const auto H =
            [&, t](const Point<2>& x)
            {
              const double T0 = x[0] / u0;
              if (T0 < t)
                return h_in + a_dot * T0;
              return H0.value(x - t * U.value(x)) + a_dot * T0;
            };

          return Fn<2>(H);
        };

      test_solver(H0, U, A, H_inflow, H);
    }
  }


  return 0;
}
