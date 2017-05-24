
/**
 * @defgroup field Field
 *
 * @brief Classes for representing spatial fields that have been discretized in
 * some finite element basis.
 *
 * Fields and vector fields are the core data types that you will be using in
 * icepack; they are the input and output types for every glacier model.
 */

/**
 * @defgroup algebra Algebra
 *
 * @brief Classes and functions for defining linear algebra operations on fields
 * and vector fields.
 *
 * For example, one would like to be able to add and subtract fields, and to
 * multiply them by scalars. Another way of saying this is that fields make up a
 * vector space, and we should be able to perform all the usual vector space
 * operations. Additionally, we can define an inner product and, in turn, a norm
 * on the space of all fields; in other words, the space of fields is a Hilbert
 * space.
 *
 * Writing code to execute linear algebra operations is certainly helpful for
 * many of the things we'd like to do with our simulation results. Unfortunately
 * it's really easy to write naive code that looks like the underlying linear
 * algebra, but that runs very slowly. Say that you were to write the following
 * expression, where `u`, `v` and `w` are fields and `alpha` and `beta` are
 * scalars:
 *
 *     Field<2> q = alpha * u - beta * (v + w);
 *
 * With a naive implementation of the algebraic operations `*`, `+` and `-`,
 * this would be translated into the following equivalent code:
 *
 *     Field<2> temp1 = v + w;
 *     Field<2> temp2 = beta * temp1;
 *     Field<2> temp3 = alpha * temp1;
 *     Field<2> q = temp2 - temp3;
 *
 * This creates three temporary objects. Moreover, the number of temporary
 * objects increases with the size of the expression. If we were to write
 *
 *     Field<2> q(u.discretization());
 *     for (size_t k = 0; k < q.coefficients().size(); ++k)
 *       q.coefficients()[k] = alpha * u.coefficient(k) -
 *         beta * (v.coefficient(k) + w.coefficient(k));
 *
 * instead then no temporary objects are created, but the code is much less
 * readable.
 *
 * We can write code that both looks like linear algebra and performs like a
 * hand-written loop using *expression templats*. Rather than have the linear
 * algebra operations return a field or vector field directly, they instead
 * return a proxy object from which one can compute the coefficients of the
 * resulting expression. The actual coefficients are not completely evaluated
 * until we assign an expression to a field -- the evaluation is delayed until
 * we know all of the ingredients that have to go into the result. The cost of
 * all this is some fairly involved template metaprogramming. If you want to
 * read more about how expression templates work, see the [Wikipedia
 * article](https://en.wikipedia.org/wiki/Expression_templates). This technique
 * has been used successfully in several C++ linear algebra libraries, chiefly
 * [Eigen](https://eigen.tuxfamily.org).
 *
 * For the most part, all of these details should be invisible to the you as a
 * user, provided I've done my job right! There is one hitch, and that relates
 * to the interaction of expression templates and C++11's automatic type
 * deduction. Since algebraic operations return proxy objects and not fields, if
 * you use `auto` to declare the type of the result of an expression, you'll get
 * a proxy object and not a field:
 *
 *     const auto q = u + v;
 *
 * In this expression, `q` will be deduced to have the type `AddExpr<...>` for
 * some appropriate template arguments; the intended result should probably be
 *
 *     const Field<2> q = u + v;
 *
 * which will force the evaluation of `u + v`. So in short, you should avoid
 * using `auto` for the result type of algebraic expressions. This is admittedly
 * a bit of a restriction. But you should already avoid using `auto` in the
 * first place when returning fields and vector fields in order to use the C++
 * type checker to catch mismatches between fields and vector fields, or to
 * catch mismatches between primal fields and dual fields.
 *
 * @ingroup field
 */

/**
 * @defgroup interpolation Interpolation
 *
 * @brief Procedures for interpolating functions to a finite element
 * discretization.
 *
 * The kinds of functions you'll want to interpolate could be defined through
 * some analytical formula, in the case of synthetic data or for a problem with
 * an exact solution. For more realistic problems, they could also come from a
 * gridded data set.
 *
 * @ingroup field
 */
