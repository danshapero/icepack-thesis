
#ifndef ICEPACK_UTILITIES_HPP
#define ICEPACK_UTILITIES_HPP

#include <deal.II/fe/fe_update_flags.h>

namespace icepack
{
  /**
   * @brief Default values for the `dealii::UpdateFlags` that are passed to
   * `dealii::FEValues`
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
