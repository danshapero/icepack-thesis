
#include <deal.II/grid/grid_refinement.h>
#include <icepack/discretization.hpp>
#include "testing.hpp"

int main(int argc, char ** argv)
{
  std::set<std::string> args = icepack::testing::get_cmdline_args(argc, argv);
  const bool refined = args.count("--refined");

  dealii::Triangulation<2> tria(icepack::testing::example_mesh(5, refined));
  const std::shared_ptr<const icepack::Discretization<2>> discretization =
      icepack::make_discretization(tria, 1);

  CHECK(discretization->triangulation().n_cells() == tria.n_cells());
  CHECK(discretization->quad().size() > 0);
  CHECK(discretization->face_quad().size() > 0);

  for (unsigned int rank = 0; rank < 2; ++rank)
  {
    const auto& rankd = discretization->operator()(rank);
    CHECK(rankd.finite_element().n_components() == std::pow(2, rank));
    CHECK(rankd.dof_handler().has_active_dofs());
    CHECK(not rankd.sparsity_pattern().empty());

    CHECK((rankd.constraints().n_constraints() == 0) xor refined);
    CHECK(rankd.make_zero_boundary_values(0).size() > 0);
    CHECK(rankd.mass_matrix().l1_norm() > 0.0);
  }

  CHECK(discretization->begin() != discretization->end());

  return 0;
}

