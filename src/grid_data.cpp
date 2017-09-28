
#include <icepack/grid_data.hpp>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <fstream>

namespace icepack
{
  using dealii::Point;
  using dealii::Table;
  using dealii::TableIndices;

  template <int dim>
  GridData<dim>::GridData(
    const Point<dim>& lower_,
    const Point<dim>& upper_,
    const Table<dim, double>& data_,
    const Table<dim, bool>& mask_
  ) :
    data(data_), mask(mask_), lower(lower_), upper(upper_)
  {
    Assert(data.size() == mask.size(),
           dealii::ExcInternalError("Table size mismatch in building "
                                    "GridData; data and mask must be the same "
                                    "size."));
  }



  namespace
  {
    double interpolate (const Table<2,double>& data,
                        const TableIndices<2>& index,
                        const Point<2>& p)
    {
      return (((1-p[0])*data[index[0]][index[1]]
               +
               p[0]*data[index[0]+1][index[1]])*(1-p[1])
              +
              ((1-p[0])*data[index[0]][index[1]+1]
               +
               p[0]*data[index[0]+1][index[1]+1])*p[1]);
    }


    template <int dim, typename T, typename F>
    void table_index_for_each(
      const Table<dim, T>& table,
      const TableIndices<dim>& index,
      F&& f
    )
    {
      for (size_t n = 0; n < (1 << dim); ++n)
      {
        dealii::TableIndices<dim> idx = index;
        for (size_t k = 0; k < dim; ++k)
          idx[k] += (bool)(n & (1 << k));

        f(table(idx));
      }
    }

  } // anonymous namespace


  template <int dim>
  boost::optional<double> GridData<dim>::operator()(const Point<dim>& x) const
  {
    if (is_masked(x))
      return boost::optional<double>{};

    const TableIndices<dim> index = table_index(x);
    const Point<dim> p = local_coords(x, index);
    return interpolate(data, index, p);
  }


  template <int dim>
  double GridData<dim>::
  value(const Point<dim>& x, const unsigned int component) const
  {
    Assert(component == 0, dealii::ExcMessage("Scalar function has only 1 component"));
    const boost::optional<double> z = (*this)(x);
    AssertThrow(z, dealii::ExcMessage("Attempt to interpolate data where gridded data"
                              " is not available."));
    return z.get();
  }


  template <int dim>
  bool GridData<dim>::is_masked(const Point<dim>& x) const
  {
    const TableIndices<dim> index = table_index(x);
    bool m = false;
    const auto f = [&m](const bool entry){ m |= entry; };
    table_index_for_each(mask, index, f);
    return m;
  }


  template <int dim>
  std::array<size_t, dim> GridData<dim>::n_subintervals() const
  {
    std::array<size_t, dim> n_subintervals;
    for (size_t k = 0; k < dim; ++k)
      n_subintervals[k] = data.size(k) - 1;
    return n_subintervals;
  }


  template <int dim>
  Point<dim> GridData<dim>::
  grid_point(const std::array<size_t, dim>& index) const
  {
    const Point<dim> dx = cell_size();
    Point<dim> x = lower;
    for (size_t k = 0; k < dim; ++k)
      x[k] += index[k] * dx[k];

    return x;
  }


  template <int dim>
  TableIndices<dim> GridData<dim>::table_index(const Point<dim>& x) const
  {
    const std::array<size_t, dim> sizes = n_subintervals();
    const Point<dim> dx = cell_size();
    TableIndices<dim> index;
    for (size_t k = 0; k < dim; ++k)
    {
      if (x[k] <= lower[k])
        index[k] = 0;
      else if (x[k] >= upper[k] - dx[k])
        index[k] = sizes[k] - 1;
      else
        index[k] = (unsigned int)((x[k] - lower[k]) / dx[k]);
    }

    return index;
  }


  template <int dim>
  Point<dim> GridData<dim>::local_coords(
    const Point<dim>& x,
    const TableIndices<dim>& index
  ) const
  {
    const Point<dim> dx = cell_size();
    Point<dim> p;
    for (size_t k = 0; k < dim; ++k)
      p[k] =
        std::max(std::min((x[k] - lower[k] - index[k] * dx[k]) / dx[k], 1.), 0.);
    return p;
  }


  template <int dim>
  Point<dim> GridData<dim>::cell_size() const
  {
    const std::array<size_t, dim> sizes = n_subintervals();
    Point<dim> p;
    for (size_t k = 0; k < dim; ++k)
      p[k] = (upper[k] - lower[k]) / sizes[k];
    return p;
  }


  template class GridData<2>;


  GridData<2> read_arc_ascii_grid(const std::string& filename)
  {
    unsigned int nx, ny;
    double x0, y0, dx, dy, missing;
    std::string dummy;

    std::ifstream file_stream(filename);
    file_stream >> dummy >> nx >> dummy >> ny;
    file_stream >> dummy >> x0 >> dummy >> y0;
    file_stream >> dummy >> dx >> dummy >> missing;
    dy = dx;

    Point<2> lower(x0, y0);
    Point<2> upper(x0 + (nx - 1) * dx, y0 + (ny - 1) * dy);

    Table<2, double> table(nx, ny);
    Table<2, bool> mask(nx, ny);

    for (unsigned int i = 0; i < ny; ++i)
      for (unsigned int j = 0; j < nx; ++j)
        file_stream >> table(j, ny - i - 1);

    file_stream.close();

    for (unsigned int i = 0; i < nx; ++i)
      for (unsigned int j = 0; j < ny; ++j)
        mask(i, j) = (table(i, j) == missing);

    return GridData<2>(lower, upper, table, mask);
  }


  void write_arc_ascii_grid(
    const GridData<2>& grid,
    const std::string& filename,
    const double no_data_value
  )
  {
    const std::array<size_t, 2> n_subintervals = grid.n_subintervals();
    const size_t nx = n_subintervals[0] + 1;
    const size_t ny = n_subintervals[1] + 1;

    const dealii::Point<2> cell_size = grid.cell_size();
    Assert(std::abs(cell_size[0] - cell_size[1]) < 1.0e-8 * cell_size.norm(),
           dealii::ExcInternalError());
    const double dx = cell_size[0];

    const dealii::Point<2> xo = grid.grid_point(std::array<size_t, 2>{{0, 0}});

    std::ofstream file_stream(filename);
    file_stream << "ncols          " << nx << "\n"
                << "nrows          " << ny << "\n"
                << "xllcorner      " << xo[0] << "\n"
                << "yllcorner      " << xo[1] << "\n"
                << "cellsize       " << dx << "\n"
                << "NODATA_value   " << no_data_value << "\n";

    for (unsigned int i = 0; i < ny; ++i)
    {
      for (unsigned int j = 0; j < nx; ++j)
      {
        const bool mask = grid.mask(j, ny - i - 1);
        file_stream << (mask ? no_data_value : grid.data(j, ny - i - 1));
        file_stream << " ";
      }

      file_stream << "\n";
    }

    file_stream.close();
  }


  namespace
  {
    template <int dim, typename F>
    void table_indices_for_each(
      const TableIndices<dim>& index1,
      const TableIndices<dim>& index2,
      F&& f
    )
    {
      size_t N = 1;
      for (size_t k = 0; k < dim; ++k)
        N *= index2[k] - index1[k];

      for (size_t n = 0; n < N; ++n)
      {
        TableIndices<dim> index;
        size_t m = n;
        for (size_t k = 0; k < dim; ++k)
        {
          index[k] = m % (index2[k] - index1[k]);
          m /= (index2[k] - index1[k]);
        }

        f(index);
      }
    }
  }


  template <int dim>
  GridData<dim> field_to_grid(const Field<dim>& phi, const double dx)
  {
    const auto& discretization = phi.discretization();
    const auto& triangulation = discretization.triangulation();

    TableIndices<dim> n_points;
    const auto predicate = [](const auto &){ return true; };
    std::pair<Point<dim>, Point<dim>> box =
      dealii::GridTools::compute_bounding_box(triangulation, predicate);
    for (size_t k = 0; k < dim; ++k)
    {
      box.first[k] -= dx;
      n_points[k] = std::ceil((box.second[k] - box.first[k]) / dx) + 1;
      box.second[k] = box.first[k] + (n_points[k] - 1) * dx;
    }

    Table<dim, double> data;
    Table<dim, bool> mask;
    data.reinit(n_points);
    mask.reinit(n_points);

    data.fill(std::numeric_limits<double>::signaling_NaN());
    mask.fill(true);

    const auto& dof_handler = discretization.scalar().dof_handler();
    const auto& coefficients = phi.coefficients();
    const dealii::Functions::FEFieldFunction<dim>
      Phi(dof_handler, coefficients);

    const auto f = [&](const TableIndices<dim>& index)
    {
      Point<dim> x = box.first;
      for (size_t k = 0; k < dim; ++k)
        x[k] += index[k] * dx;

      try
      {
        const double z = Phi.value(x);
        mask(index) = false;
        data(index) = z;
      }
      catch (...)
      {}
    };

    table_indices_for_each(TableIndices<dim>(), n_points, f);
    return GridData<dim>(box.first, box.second, data, mask);
  }

  template GridData<2> field_to_grid(const Field<2>&, const double);
}

