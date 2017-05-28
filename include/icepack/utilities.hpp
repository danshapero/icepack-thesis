
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


  /**
   * Return `true` if the given face of a deal.II cell iterator is on the
   * boundary of the mesh.
   */
  template <typename iterator>
  bool at_boundary(const iterator& it, const unsigned int face_number)
  {
    return it->face(face_number)->at_boundary();
  }


  /**
   * Return `true` if the given face of a deal.II iterator is on a specific
   * part of the boundary of the mesh.
   */
  template <typename iterator>
  bool at_boundary(
    const iterator& it,
    const unsigned int face_number,
    const unsigned int boundary_id
  )
  {
    return at_boundary(it, face_number) and
      (it->face(face_number)->boundary_id() == boundary_id);
  }

}

#endif
