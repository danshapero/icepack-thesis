
#include <icepack/mesh.hpp>
#include <deal.II/grid/grid_in.h>
#include <fstream>

namespace icepack
{

  dealii::Triangulation<2> read_msh(const std::string& filename)
  {
    dealii::GridIn<2> grid_in;
    dealii::Triangulation<2> tria;
    grid_in.attach_triangulation(tria);
    std::ifstream stream(filename);
    grid_in.read_msh(stream);
    return tria;
  }

}

