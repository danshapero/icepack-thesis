
#ifndef ICEPACK_MASS_TRANSPORT_HPP
#define ICEPACK_MASS_TRANSPORT_HPP

#include <icepack/field.hpp>

namespace icepack
{
  /**
   * @brief Return the timestep `dt` that gives a Courant number of a given
   * value for some velocity field.
   *
   * The Courant number for the advection equation with a velocity \f$u\f$,
   * mesh spacing \f$\delta x\f$, and timestep \f$\delta t\f$ is the quantity
   * \f[
   *   C = \max|u|\cdot \delta t/\delta x.
   * \f]
   * Many numerical methods for solving the advection equation are only stable
   * when the Courant number is below some upper limit.
   */
  double compute_timestep(
    const double courant_number,
    const VectorField<2>& u
  );


  /**
   * @brief Updates the ice thickness from the continuity equation using an
   * implicit timestepping scheme
   *
   * The ice thickness \f$h\f$ evolves according to the advection equation
   * \f[
   *   \frac{\partial h}{\partial t} + \nabla\cdot hu = \dot a - \dot m
   * \f]
   * where \f$u\f$ is the ice velocity, \f$\dot a\f$ is the accumulation rate
   * and \f$m\f$ is the melt rate.
   *
   * @ingroup physics
   */
  struct MassTransport
  {
    MassTransport();

    /**
     * Compute the matrix discretizing the action of the advective flux on a
     * given thickness field
     */
    dealii::SparseMatrix<double> flux_matrix(
      const VectorField<2>& velocity,
      const dealii::ConstraintMatrix& constraints = dealii::ConstraintMatrix()
    ) const;

    /**
     * Implementation of the mass transport solve using an implicit time
     * discretization
     */
    Field<2> solve(
      const double dt,
      const Field<2>& thickness,
      const Field<2>& accumulation,
      const VectorField<2>& velocity,
      const std::set<dealii::types::boundary_id>& inflow_boundary_ids,
      const Field<2>& inflow_thickness
    ) const;
  };

}

#endif

