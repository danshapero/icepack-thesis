
#include <deal.II/fe/fe_values.h>
#include <icepack/field.hpp>
#include "testing.hpp"

using icepack::Field;
using icepack::VectorField;
using icepack::DualField;
using icepack::DualVectorField;

const auto update_flags =
  dealii::update_values | dealii::update_quadrature_points;


/// This function takes in a field or vector field and some function-like
/// object, evaluates the difference between the field and the function at every
/// quadrature point of every cell of the underlying mesh, and returns the max
/// discrepancy between the two.
template <int rank, int dim, class F>
double diff(const icepack::FieldType<rank, dim>& u, const F& f)
{
  const auto& discretization = u.discretization();
  const auto& fe = discretization(rank).finite_element();
  const auto quad = discretization.quad();
  const auto n_q_points = quad.size();
  using value_type = typename icepack::FieldType<rank, dim>::value_type;
  std::vector<value_type> u_values(n_q_points);
  std::vector<value_type> f_values(n_q_points);

  dealii::FEValues<dim> fe_values(fe, quad, update_flags);
  const typename icepack::FieldType<rank, dim>::extractor_type ex(0);

  double d = 0.0;
  for (auto cell: discretization(rank).dof_handler().active_cell_iterators())
  {
    fe_values.reinit(cell);

    f.value_list(fe_values.get_quadrature_points(), f_values);
    fe_values[ex].get_function_values(u.coefficients(), u_values);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const value_type diff = u_values[q] - f_values[q];
      d = std::max(d, std::sqrt(diff * diff));
    }
  }

  return d;
}


#define CHECK_FUNC(phi, Phi, tolerance)         \
  CHECK_REAL(diff((phi), (Phi)), 0.0, (tolerance));



using icepack::testing::AffineFunction;
using icepack::testing::AffineTensorFunction;

int main()
{
  const dealii::Triangulation<2> tria = icepack::testing::example_mesh();
  const icepack::Discretization<2> discretization(tria, 1);

  const double tolerance = std::pow(icepack::testing::resolution(tria), 2);

  const AffineFunction Phi(1.0, dealii::Point<2>(-4.0, 8.0));
  const AffineFunction Psi(-2.0, dealii::Point<2>(3.0, 7.0));
  const AffineTensorFunction U(Phi, Psi);
  const AffineFunction Xi(-6.0, dealii::Point<2>(1.3, 0.0));
  const AffineFunction Chi(0.0, dealii::Point<2>(-1.0, 8.6));
  const AffineTensorFunction V(Xi, Chi);
  const AffineTensorFunction W(Psi, Xi);


  TEST_SUITE("basic field operations")
  {
    const auto test = [&](const auto& Phi, const auto& Psi)
    {
      const auto phi = icepack::interpolate(discretization, Phi);
      CHECK_FUNC(phi, Phi, tolerance);
      CHECK_REAL(norm(phi), norm(Phi), tolerance);

      const auto psi = icepack::interpolate(discretization, Psi);
      CHECK_REAL(inner_product(phi, psi),
                 icepack::testing::inner_product(Phi, Psi),
                 tolerance);

      CHECK_REAL(dist(phi, psi), norm(Phi - Psi), tolerance);

      const auto f = transpose(phi);
      CHECK_REAL(inner_product(f, psi), inner_product(phi, psi), tolerance);
      CHECK_FIELDS(transpose(f), phi, tolerance);
    };

    test(Phi, Psi);
    test(U, V);
  }


  TEST_SUITE("copying fields")
  {
    const auto test = [&](const auto& Phi, const auto& Psi)
    {
      auto phi = icepack::interpolate(discretization, Phi);
      auto psi(phi);
      CHECK_FIELDS(phi, psi, tolerance);

      psi = icepack::interpolate(discretization, Psi);
      phi = psi;
      CHECK_FIELDS(phi, psi, tolerance);

      auto f = transpose(phi);
      auto g(f);
      CHECK_REAL(inner_product(f, psi), inner_product(g, psi), tolerance);

      f = transpose(psi);
      g = f;
      CHECK_REAL(inner_product(f, phi), inner_product(g, phi), tolerance);
    };

    test(Phi, Psi);
    test(U, V);
  }


  TEST_SUITE("moving fields")
  {
    const auto test = [&](const auto& Phi)
    {
      auto phi = icepack::interpolate(discretization, Phi);
      auto psi(std::move(phi));
      CHECK(phi.coefficients().size() == 0);
      CHECK(psi.coefficients().size() > 0);

      phi = std::move(psi);
      CHECK(phi.coefficients().size() > 0);
      CHECK(psi.coefficients().size() == 0);

      auto f = transpose(phi);
      auto g(std::move(f));
      CHECK(f.coefficients().size() == 0);
      CHECK(g.coefficients().size() > 0);

      f = std::move(g);
      CHECK(f.coefficients().size() > 0);
      CHECK(g.coefficients().size() == 0);
    };

    test(Phi);
    test(U);
  }


  TEST_SUITE("algebra on fields")
  {
    const auto test = [&](const auto& Phi, const auto& Psi)
    {
      auto phi = icepack::interpolate(discretization, Phi);
      phi *= 2;
      CHECK_FUNC(phi, 2 * Phi, tolerance);

      auto psi = icepack::interpolate(discretization, Psi);
      phi += psi;
      CHECK_FUNC(phi, 2 * Phi + Psi, tolerance);
    };

    test(Phi, Psi);
    test(U, V);
  }


  TEST_SUITE("field expression templates")
  {
    const auto test = [&](const auto& Phi1, const auto& Phi2, const auto& Phi3)
    {
      auto phi1 = icepack::interpolate(discretization, Phi1);
      auto phi2 = icepack::interpolate(discretization, Phi2);
      auto phi3 = icepack::interpolate(discretization, Phi3);

      decltype(phi1) phi = phi1 - phi3;
      CHECK_FUNC(phi, Phi1 - Phi3, tolerance);

      phi = phi1 + phi2;
      CHECK_FUNC(phi, Phi1 + Phi2, tolerance);

      phi = 2 * phi1 + phi2;
      CHECK_FUNC(phi, 2 * Phi1 + Phi2, tolerance);

      phi = phi1 + 3 * phi2 + phi3;
      CHECK_FUNC(phi, Phi1 + 3 * Phi2 + Phi3, tolerance);

      phi = 4 * phi1 + 7 * phi2 + 3 * phi3;
      CHECK_FUNC(phi, 4 * Phi1 + 7 * Phi2 + 3 * Phi3, tolerance);

      phi = phi1 - phi2;
      CHECK_FUNC(phi, Phi1 - Phi2, tolerance);

      phi = phi1;
      phi += 2 * phi2;
      phi -= 3 * phi3;
      CHECK_FUNC(phi, Phi1 + 2 * Phi2 - 3 * Phi3, tolerance);
    };

    test(Phi, Psi, Xi);
    test(U, V, W);
  }


  return 0;
}
