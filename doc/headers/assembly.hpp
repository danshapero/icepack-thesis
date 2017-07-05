
/**
 * @defgroup assembly Assembly
 *
 * @brief Functionality for assembly of scalar, vectors, or matrices over the
 * cells of a triangulation.
 *
 * Most of the interesting quantities you might wish to compute in finite
 * element analysis are obtained by summing over the cells of a triangulation.
 * The setup process for these computations is very similar, no matter what it
 * is that you're trying to compute:
 *
 *   1. Retrieve the `Discretization` object from the input fields and vector
 *   fields.
 *   2. The discretization constructs a `dealii::Quadrature` object, which we
 *   can use to approximate an integrals in the unit cube
 *   3. Knowing the size of the quadrature rule, we make some `std::vector`s to
 *   store the values, gradients, etc. of each field at the quadrature points
 *   of the current cell.
 *   4. The discretization is used to construct two `dealii::FEValues` objects,
 *   one for scalar fields and one for vector fields; these are used to update
 *   the field values on each cell.
 *   5. Loop over all the cells of the triangulation.
 *   6. Update the `dealii::FEValues` objects from the current cell.
 *   7. The `dealii::FEValues` objects now update the vectors of field values
 *   for the current cell.
 *   8. Loop over all the quadrature points for the unit cell.
 *   9. Compute the value of the integrand at each quadrature point and add it
 *   to the cell-wise sum.
 *   10. Add the cell-wise sum to the global sum.
 *
 * Of all of these steps, only 3, 7, and 9 actually differ depending on what it
 * is you're trying to compute. The classes and functions in this group are for
 * automating everything else.
 *
 * The main classes for assisting with assembly are:
 *
 *   * `Evaluator`: keeps track of cell-wise data (values, gradients, etc.) of
 *   a single scalar or vector field
 *   * `evaluate`: family of functions for creating `Evaluator` objects for
 *   each type of quantity you might wish to evaluate (values, gradients, etc.)
 *   for a field
 *   * `AssemblyData`: aggregates several `Evaluator`s together, along with
 *   `dealii::FEValues` objects for scalar and vector fields; these FE values
 *   objects are used to update the evaluators
 *   * `AssemblyFaceData`: same as `AssemblyData` but for assembly over the
 *   faces of a mesh, with `dealii::FEFaceValues` instead of `dealii::FEValues`
 *
 * The functions `make_assembly_data`, `make_assembly_face_data` are used to
 * construct respectively `AssemblyData` and `AssemblyFaceData` objects.
 *
 * A brief code snippet showing how these classes are used:
 *
 * @code{.cpp}
 *   icepack::Field<2> h = ...;
 *   icepack::VectorField<2> u = ...;
 *
 *   const auto assembly_data =
 *     icepack::make_assembly_data<2>(
 *       icepack::evaluate::function(h),
 *       icepack::evaluate::symmetric_gradient(u)
 *     );
 *
 *  const auto face_assembly_data =
 *    icepack::make_assembly_face_data<2>(
 *      icepack::evaluate::function(h),
 *      icepack::evaluate::function(u)
 *    );
 * @endcode
 *
 * In this example, the results from the `evaluate` functions were passed
 * directly to the functions to create assembly data; we never actually stored
 * the `Evaluator`s anywhere. Additionally, whether you're making cell or face
 * assembly data, it's still the same call to `evaluate::function`,
 * `evaluate::gradient`, etc. -- the functions that make the assembly data for
 * you figure out how much space to allocate for cell values.
 */

