
#ifndef ICEPACK_SEQUENCE_HPP
#define ICEPACK_SEQUENCE_HPP

#include <vector>

namespace icepack
{
  namespace numerics
  {
    /**
     * @brief Class template for representing recursively-defined sequences
     *
     * Many iterative methods are expressed as some short recurrence. For
     * example, the next search direction in the BFGS algorithm is defined in
     * terms of the last few guesses and the gradient of the objective
     * functional at these guesses. This class template is for making the code
     * for these recurrences look as much like the underlying math as possible.
     *
     * The `Sequence` class template presents an interface that looks like that
     * of a hypothetically infinite vector. Accessing element `n` actually
     * accesses element `n % N` of the underlying data vector, the idea being
     * that an expression of the form
     *
     *     s[n] = f(s[n - 1], s[n - 2], ..., s[n - N]);
     *
     * will access the previous `N` entries of the sequence to define the next
     * entry. In principle, you could do something goofy like try to access a
     * really "old" entry and get a recent one instead; there are no safeguards
     * against that.
     *
     * TODO: Write safeguards to check that all accesses are in the correct
     * bounds.
     */
    template <typename T>
    class Sequence
    {
    public:
      /**
       * Construct a new sequence of a given length. The initial contents of
       * the sequence are uninitialized.
       */
      Sequence(const size_t length);

      /**
       * Return the recurrence length of the sequence.
       */
      size_t length() const;

      /**
       * Return const access to a given member of the sequence.
       */
      const T& operator[](const size_t n) const;

      /**
       * Return mutable access to a given member of the sequence.
       */
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
