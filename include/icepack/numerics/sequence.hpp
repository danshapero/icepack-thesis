
#ifndef ICEPACK_SEQUENCE_HPP
#define ICEPACK_SEQUENCE_HPP

#include <vector>

namespace icepack
{
  namespace numerics
  {

    /**
     * This class is used to represent short recurring sequences, such as a
     * sequence of approximations in some iterative method. These methods
     * usually define the next element of the sequence in terms of the last
     * `N` elements. For example, the next search direction in the BFGS
     * algorithm is defined in terms of the last few guesses and the gradient
     * of the objective functional at these guesses.
     */
    template <typename T>
    class Sequence
    {
    public:
      Sequence(const size_t length);

      size_t length() const;

      const T& operator[](const size_t n) const;

      T& operator[](const size_t n);

    protected:
      std::vector<T> elements;
    };


    template <typename T>
    Sequence<T>::Sequence(const size_t length) :
      elements(length)
    {}


    template <typename T>
    size_t Sequence<T>::length() const
    {
      return elements.size();
    }


    template <typename T>
    const T& Sequence<T>::operator[](const size_t n) const
    {
      return elements[n % elements.size()];
    }


    template <typename T>
    T& Sequence<T>::operator[](const size_t n)
    {
      return elements[n % elements.size()];
    }

  }
}

#endif
