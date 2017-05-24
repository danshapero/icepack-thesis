
namespace icepack
{
  /** @namespace icepack::numerics
   *
   * @brief Functions and classes for numerical algorithms, especially for
   * convex optimization.
   *
   * Usually the velocity of a glacier is specified as the solution of some
   * nonlinear elliptic system of PDEs. These PDEs can in turn be derived from
   * an *action principle*. An action principle states that the solution of a
   * PDE arises from the extremization of a scalar functional; the PDE is the
   * derivative of the action functional. All of the common glacier models (the
   * full Stokes equations, Blatter-Pattyn, shallow shelf, etc.) have an action
   * functional, and moreover each of these action functionals is strictly
   * *convex*. Convexity guarantees that the action has a unique minimizer. We
   * can leverage these action principles to make much more robust solvers for
   * the model equations than might otherwise be possible if we only considered
   * them as a large system of equations. The procedures in this namespace are
   * used for solving these kinds of convex optimization problems.
   */
  namespace numerics;
}
