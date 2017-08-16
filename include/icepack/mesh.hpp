
#ifndef ICEPACK_MESH_HPP
#define ICEPACK_MESH_HPP

#include <deal.II/grid/tria.h>

namespace icepack
{
  /**
   * @brief Read a gmsh `.msh` file into a deal.II triangulation
   *
   * The program [gmsh](http://gmsh.info) generates high-quality unstructured
   * quadrilateral meshes, which are exactly what we need for working with
   * deal.II. This function wraps some of deal.II's built-in functionality for
   * reading meshes in the `.msh` format that gmsh generates.
   */
  dealii::Triangulation<2> read_msh(const std::string& filename);
}

#endif

