
#include "testing.hpp"
#include <deal.II/lac/vector.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

using dealii::Point;
using dealii::Tensor;

namespace icepack
{
  namespace testing
  {

    std::set<std::string> get_cmdline_args(int argc, char ** argv)
    {
      std::set<std::string> args;
      for (int k = 0; k < argc; ++k)
        args.insert(std::string(argv[k]));

      return args;
    }


    dealii::Triangulation<2>
    example_mesh(unsigned int num_levels, bool refined)
    {
      dealii::Triangulation<2> tria;
      dealii::GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
      tria.refine_global(num_levels);

      if (refined)
      {
        dealii::Vector<double> refinement_criteria(tria.n_active_cells());
        for (const auto cell: tria.active_cell_iterators())
        {
          const unsigned int index = cell->index();
          dealii::Point<2> x = cell->barycenter();
          refinement_criteria[index] = x[0];
        }

        dealii::GridRefinement::refine(tria, refinement_criteria, 0.5);
        tria.execute_coarsening_and_refinement();
      }

      return tria;
    }


    dealii::Triangulation<2>
    rectangular_mesh(
      double length, double width, unsigned int num_levels, bool refined
    )
    {
      dealii::Triangulation<2> tria = example_mesh(num_levels, refined);

      const auto f = [=](const dealii::Point<2>& x)
      {
        return dealii::Point<2>(length * x[0], width * x[1]);
      };

      dealii::GridTools::transform(f, tria);
      return tria;
    }


    template <int dim>
    double resolution(const dealii::Triangulation<dim>& tria)
    {
      return
        dealii::GridTools::minimal_cell_diameter(tria) /
        dealii::GridTools::diameter(tria);
    }

    template double resolution<2>(const dealii::Triangulation<2>&);


    bool is_decreasing(const std::vector<double>& seq)
    {
      for (size_t k = 1; k < seq.size(); ++k)
        if (seq[k] > seq[k - 1])
          return false;

      return true;
    }


    AffineFunction::AffineFunction(double a_, const dealii::Tensor<1, 2>& p_) :
      a(a_), p(p_)
    {}


    double inner_product(const AffineFunction& f, const AffineFunction& g)
    {
      return f.a * g.a
        + f.a * (g.p[0] + g.p[1]) / 2
        + g.a * (f.p[0] + f.p[1]) / 2
        + (f.p[0] * g.p[1] + f.p[1] * g.p[0]) / 4
        + (f.p * g.p) / 3;
    }


    double norm(const AffineFunction& f)
    {
      return std::sqrt(inner_product(f, f));
    }


    double max(const AffineFunction& f, const dealii::Triangulation<2>& tria)
    {
      double max_val = 0.0;
      for (const Point<2>& x: tria.get_vertices())
        max_val = std::max(max_val, std::abs(f.value(x)));

      return max_val;
    }


    double AffineFunction::value(const Point<2>& x, const unsigned int) const
    {
      return a + p * x;
    }


    AffineFunction operator+(const AffineFunction& f, const AffineFunction& g)
    {
      return AffineFunction(f.a + g.a, f.p + g.p);
    }


    AffineFunction operator-(const AffineFunction& f, const AffineFunction& g)
    {
      return AffineFunction(f.a - g.a, f.p - g.p);
    }


    AffineFunction operator*(double alpha, const AffineFunction& f)
    {
      return AffineFunction(alpha * f.a, alpha * f.p);
    }


    AffineTensorFunction::
    AffineTensorFunction(const AffineFunction& ux, const AffineFunction& uy) :
      coords{{ux, uy}}
    {}

    Tensor<1, 2> AffineTensorFunction::value(const Point<2>& x) const
    {
      return Tensor<1, 2>{{coords[0].value(x), coords[1].value(x)}};
    }


    AffineTensorFunction
    operator+(const AffineTensorFunction& f, const AffineTensorFunction& g)
    {
      return AffineTensorFunction(f.coords[0] + g.coords[0],
                                  f.coords[1] + g.coords[1]);
    }

    AffineTensorFunction
    operator-(const AffineTensorFunction& f, const AffineTensorFunction& g)
    {
      return AffineTensorFunction(f.coords[0] - g.coords[0],
                                  f.coords[1] - g.coords[1]);
    }

    AffineTensorFunction operator*(double alpha, const AffineTensorFunction& f)
    {
      return AffineTensorFunction(alpha * f.coords[0], alpha * f.coords[1]);
    }

    double
    inner_product(const AffineTensorFunction& f, const AffineTensorFunction& g)
    {
      return inner_product(f.coords[0], g.coords[0])
        + inner_product(f.coords[1], g.coords[1]);
    }

    double norm(const AffineTensorFunction& f)
    {
      return std::sqrt(inner_product(f, f));
    }


    double
    max(const AffineTensorFunction& f, const dealii::Triangulation<2>& tria)
    {
      double max_val = 0.0;
      for (const Point<2>& x: tria.get_vertices())
        max_val = std::max(max_val, f.value(x).norm());

      return max_val;
    }

  } // End of namespace testing
} // End of namespace icepack
