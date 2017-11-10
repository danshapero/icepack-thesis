
#include <icepack/inverse/error_functionals.hpp>
#include "testing.hpp"

using dealii::Point;
using dealii::Tensor;
using icepack::testing::Fn;
using icepack::testing::TensorFn;

int main(int argc, char ** argv)
{
  const auto args = icepack::testing::get_cmdline_args(argc, argv);
  const bool refined = args.count("--refined");
  const bool quadratic = args.count("--quadratic");

  const size_t num_levels = 5 - quadratic;
  const size_t p = 1 + quadratic;

  const double L = 1.0;
  const double W = 1.0;

  const Fn<2> Sigma([&](const Point<2>&){return 1.0;});
  const TensorFn<2> U([&](const Point<2>&){return Tensor<1, 2>{{1.0, 0.0}};});
  const TensorFn<2> V([&](const Point<2>&){return Tensor<1, 2>{{0.0, 1.0}};});

  const dealii::Triangulation<2> tria =
    icepack::testing::rectangular_mesh(L, W, num_levels, refined);
  const auto discretization = icepack::make_discretization(tria, p);
  const double tolerance = std::pow(icepack::testing::resolution(tria), p + 1);

  const icepack::Field<2> sigma = interpolate(*discretization, Sigma);
  const icepack::VectorField<2> u = interpolate(*discretization, U);
  const icepack::VectorField<2> v = interpolate(*discretization, V);

  const icepack::MeanSquareError<2> mean_square_error{};

  CHECK_REAL(mean_square_error.action(u, v, sigma), L * W, tolerance);

  const icepack::DualVectorField<2> f =
    mean_square_error.derivative(u, v, sigma);
  const TensorFn<2>
    Q([&](const Point<2>& x){return Tensor<1, 2>{{x[0], -x[1]}};});
  const icepack::VectorField<2> q = interpolate(*discretization, Q);
  CHECK_REAL(inner_product(f, q), 1.0, tolerance);

  return 0;
}
