
#ifndef ICEPACK_UTILITIES_HPP
#define ICEPACK_UTILITIES_HPP

#include <deal.II/fe/fe_update_flags.h>

namespace icepack
{
  /**
   * This namespace contains sensible default values for the `UpdateFlags`
   * that have to be passed to `FEValues` when evaluating fields at all the
   * quadrature points of cells of a triangulation.
   */
  namespace DefaultFlags
  {
    using dealii::UpdateFlags;

    /**
     * Update flags suitable for iterating over the interior cells of a mesh.
     */
    extern const UpdateFlags flags;

    /**
     * Update flags suitable for iterating over the boundary cells of a mesh.
     */
    extern const UpdateFlags face_flags;
  }
}

#endif
