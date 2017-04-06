
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_values.h>
#include <icepack/field.hpp>
#include "testing.hpp"

using dealii::Point;
using dealii::Tensor;
using dealii::Function;
using dealii::TensorFunction;
using dealii::QGauss;
using dealii::FiniteElement;
using dealii::FEValues;
using dealii::GridTools::minimal_cell_diameter;
using dealii::GridTools::diameter;
using icepack::Discretization;
using icepack::Field;
using icepack::VectorField;
using icepack::DualField;
using icepack::DualVectorField;

const auto update_flags =
  dealii::update_values | dealii::update_quadrature_points;


template <int dim>
class ExampleField : public Function<dim>
{
  double value(const Point<dim>& x, const unsigned int = 0) const
  {
    return x[0] * x[1];
  }
};


template <int dim>
class ExampleVectorField : public TensorFunction<1, dim>
{
  Tensor<1, dim> value(const Point<dim>& x) const
  {
    Tensor<1, dim> v;
    for (unsigned int i = 0; i < dim; ++i)
      v[i] = x[(i + 1) % dim];
    return v;
  }
};


template <int dim>
void test_field(const Discretization<dim>& dsc)
{
  const ExampleField<dim> function;

  const auto& tria = dsc.triangulation();
  const double dx = minimal_cell_diameter(tria) / diameter(tria);

  Field<dim> phi = icepack::interpolate(dsc, function);

  const QGauss<dim> quad = dsc.quad();
  const unsigned int n_q_points = quad.size();

  std::vector<double> function_values(n_q_points);
  std::vector<double> phi_values(n_q_points);

  FEValues<dim> fe_values(dsc.scalar().finite_element(), quad, update_flags);
  const dealii::FEValuesExtractors::Scalar ex(0);

  for (auto cell: dsc.scalar().dof_handler().active_cell_iterators())
  {
    fe_values.reinit(cell);

    function.value_list(fe_values.get_quadrature_points(), function_values);
    fe_values[ex].get_function_values(phi.coefficients(), phi_values);

    for (unsigned int q = 0; q < n_q_points; ++q)
      check_real(phi_values[q], function_values[q], dx * dx);
  }

  const double n = norm(phi);
  const double exact_integral = 1.0/3;
  check_real(n, exact_integral, dx*dx);

  const DualField<dim> f = transpose(phi);
  check_real(inner_product(phi, phi), inner_product(f, phi), dx*dx);
  check_real(dist(transpose(f), phi), 0.0, dx*dx);
}


template <int dim>
void test_vector_field(const Discretization<dim>& dsc)
{
  const ExampleVectorField<dim> function;

  const auto& tria = dsc.triangulation();
  const double dx = minimal_cell_diameter(tria) / diameter(tria);

  VectorField<dim> u = icepack::interpolate(dsc, function);

  const QGauss<dim> quad = dsc.quad();
  const unsigned int n_q_points = quad.size();

  std::vector<Tensor<1, dim>> function_values(n_q_points);
  std::vector<Tensor<1, dim>> u_values(n_q_points);

  FEValues<dim> fe_values(dsc.vector().finite_element(), quad, update_flags);
  const dealii::FEValuesExtractors::Vector ex(0);

  for (auto cell: dsc.vector().dof_handler().active_cell_iterators())
  {
    fe_values.reinit(cell);

    function.value_list(fe_values.get_quadrature_points(), function_values);
    fe_values[ex].get_function_values(u.coefficients(), u_values);

    for (unsigned int q = 0; q < n_q_points; ++q)
      check((function_values[q] - u_values[q]).norm() < dx*dx);
  }

  const double n = norm(u);
  const double exact_integral = std::sqrt(dim/3.0);
  check_real(n, exact_integral, dx);

  const DualVectorField<2> f = transpose(u);
  check_real(inner_product(u, u), inner_product(f, u), dx*dx);
  check_real(dist(transpose(f), u), 0.0, dx*dx);
}


int main()
{
  const double length = 1.0;
  const double width = 1.0;
  auto tria = icepack::testing::rectangular_glacier(length, width);
  const Discretization<2> discretization(tria, 2);

  Field<2> phi(discretization);
  VectorField<2> v(discretization);

  const auto& dof_handler = phi.discretization().scalar().dof_handler();
  check(dof_handler.has_active_dofs());
  check(phi.coefficients().size() == dof_handler.n_dofs());
  check(v.coefficients().size() == 2 * phi.coefficients().size());

  test_field(discretization);
  test_vector_field(discretization);

  return 0;
}
