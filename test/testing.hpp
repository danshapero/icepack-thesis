
#ifndef ICEPACK_TESTING_HPP
#define ICEPACK_TESTING_HPP

#include <set>
#include <string>

#include <deal.II/grid/tria.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

namespace icepack
{
  namespace testing
  {

    /// Return a set of strings of all the command line arguments passed to a
    /// program.
    std::set<std::string> get_cmdline_args(int argc, char ** argv);


    /// Print out a vector of errors, separated by spaces
    void print_errors(const std::vector<double>& errors);


    /// Return a mesh of the unit square with a given number of refinement
    /// levels. If the optional argument `refined` is true, then the right half
    /// of the mesh will be refined; this is for testing functions on non-
    /// uniform meshes. The boundary ID of the surfaces `x = 1` will be set to 1
    /// for testing problems with non-uniform boundary conditions, all other
    /// boundaries will have ID 0.
    dealii::Triangulation<2>
    example_mesh(unsigned int num_levels = 5, bool refined = false);


    /// Return a mesh of the rectangle of the given length and width.
    dealii::Triangulation<2>
    rectangular_mesh(
      double length,
      double width,
      unsigned int num_levels = 5,
      bool refined = false
    );


    /// Return the resolution of a mesh; we use this to set the tolerance for
    /// checks when testing whether two floating-point numbers are close enough.
    template <int dim>
    double resolution(const dealii::Triangulation<dim>& tria);


    /// Check if a vector is decreasing. This function is used when we collect
    /// the errors in some approximation into a vector, and we want to make sure
    /// that the errors are decreasing to 0.
    bool is_decreasing(const std::vector<double>& seq);


    /// Return a deal.II function object given a lambda or a `std::function`.
    template <int dim>
    using Fn = dealii::ScalarFunctionFromFunctionObject<dim>;

    /// Return a deal.II TensorFunction given a lambda or a `std::function`
    template <int dim>
    class TensorFn : public dealii::TensorFunction<1, dim>
    {
    public:
      using func_type =
        std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&)>;

      TensorFn(const func_type& f) : f_(f)
      {}

      dealii::Tensor<1, dim> value(const dealii::Point<dim>& x) const
      {
        return f_(x);
      }

      const func_type f_;
    };



    /// This class is used to represent affine functions, which we use for
    /// testing a lot of the routines since (1) they can be interpolated exactly
    /// and (2) most things you'd like to compute can be done analytically.
    class AffineFunction : public dealii::Function<2>
    {
    public:
      const double a;
      const dealii::Tensor<1, 2> p;

      AffineFunction(double a, const dealii::Tensor<1, 2>& p);
      double value(const dealii::Point<2>& x, const unsigned int = 0) const;
    };


    /// Some algebra on affine functions.
    AffineFunction operator+(const AffineFunction& f, const AffineFunction& g);

    AffineFunction operator-(const AffineFunction& f, const AffineFunction& g);

    AffineFunction operator*(double alpha, const AffineFunction& f);

    double inner_product(const AffineFunction& f, const AffineFunction& g);

    double norm(const AffineFunction& f);

    double max(const AffineFunction& f, const dealii::Triangulation<2>& tria);


    /// Same as affine functions above but for vector fields.
    class AffineTensorFunction : public dealii::TensorFunction<1, 2>
    {
    public:
      std::array<AffineFunction, 2> coords;

      AffineTensorFunction(const AffineFunction& ux, const AffineFunction& uy);
      dealii::Tensor<1, 2> value(const dealii::Point<2>& x) const;
    };

    AffineTensorFunction
    operator+(const AffineTensorFunction& f, const AffineTensorFunction& g);

    AffineTensorFunction
    operator-(const AffineTensorFunction& f, const AffineTensorFunction& g);

    AffineTensorFunction operator*(double alpha, const AffineTensorFunction& f);

    double
    inner_product(const AffineTensorFunction& f, const AffineTensorFunction& g);

    double norm(const AffineTensorFunction& f);

    double
    max(const AffineTensorFunction& f, const dealii::Triangulation<2>& tria);

  } // namespace testing
} // namespace icepack


/**
 * Macros for checking errors.
 */

#include <cassert>

#define CHECK(cond)                                                     \
  do                                                                    \
  {                                                                     \
    if (!(cond))                                                        \
    {                                                                   \
      std::cerr << "Test " << #cond << std::endl                        \
                << "at   " << __FILE__ << ":" << __LINE__ << std::endl  \
                << "failed." << std::endl;                              \
      abort();                                                          \
    }                                                                   \
  } while(0)

#define CHECK_REAL(val1, val2, tol)                                     \
  do                                                                    \
  {                                                                     \
    const double __icepack_test_diff = std::abs((val1) - (val2));       \
    if (__icepack_test_diff > (tol))                                    \
    {                                                                   \
      std::cerr << "|" << #val1 << " - " << #val2 << "| = "             \
                << __icepack_test_diff << std::endl                     \
                << "  > " << #tol << " = " << (tol) << std::endl        \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;   \
      abort();                                                          \
    }                                                                   \
  } while(0)

#define CHECK_FIELDS(phi1, phi2, tol)                                   \
  do                                                                    \
  {                                                                     \
    const double __icepack_test_diff =                                  \
      icepack::dist((phi1), (phi2)) / norm(phi2);                       \
    if (__icepack_test_diff > (tol))                                    \
    {                                                                   \
      std::cerr << "||" << #phi1 << " - " << #phi2 << "|| "             \
                << "/ ||" << #phi2 << "|| = "                           \
                << __icepack_test_diff << std::endl                     \
                << "   > " << #tol << " = " << (tol) << std::endl       \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;   \
      abort();                                                          \
    }                                                                   \
  } while(0)


#define TEST_SUITE(name)                        \
  std::cout << "-- " << name << std::endl;




#endif
