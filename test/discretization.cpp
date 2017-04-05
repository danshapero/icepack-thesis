
#include <deal.II/grid/grid_refinement.h>
#include <icepack/discretization.hpp>
#include "testing.hpp"

int main(int argc, char ** argv)
{
  std::set<std::string> args = icepack::testing::get_cmdline_args(argc, argv);
  const bool refined = args.count("--refined");

  const double width = 100.0;
  const double length = 100.0;
  auto tria = icepack::testing::rectangular_glacier(width, length);

  if (refined)
  {
    dealii::Vector<double> refinement_criteria(tria.n_active_cells());
    for (const auto cell: tria.cell_iterators())
    {
      const unsigned int index = cell->index();
      dealii::Point<2> x = cell->barycenter();
      refinement_criteria[index] = x[0] / length;
    }

    dealii::GridRefinement::refine(tria, refinement_criteria, 0.5);
    tria.execute_coarsening_and_refinement();
  }

  const icepack::Discretization<2> discretization(tria, 1);

  check(discretization.triangulation().n_cells() == tria.n_cells());
  check(discretization.quad().size() > 0);

  const auto& scalar = discretization.scalar();
  const auto& vector = discretization.vector();

  check(scalar.finite_element().n_components() == 1);
  check(vector.finite_element().n_components() == 2);

  check(scalar.dof_handler().has_active_dofs());
  check(vector.dof_handler().has_active_dofs());

  check(not scalar.sparsity_pattern().empty());
  check(not vector.sparsity_pattern().empty());

  check((scalar.constraints().n_constraints() == 0) xor refined);
  check((vector.constraints().n_constraints() == 0) xor refined);

  check(scalar.mass_matrix().l1_norm() > 0.0);
  check(vector.mass_matrix().l1_norm() > 0.0);

  return 0;
}
