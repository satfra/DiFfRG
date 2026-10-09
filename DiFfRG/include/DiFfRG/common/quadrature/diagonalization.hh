#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>

// C++ standard library
#include <cmath>
#include <limits>

namespace DiFfRG
{
  /**
   * @brief Diagonalizes a symmetric tridiagonal matrix.
   *
   * Adapted from https://people.math.sc.edu/burkardt/cpp_src/cpp_src.html
   *
   * This routine is a slightly modified version of the EISPACK routine to
   * perform the implicit QL algorithm on a symmetric tridiagonal matrix.
   * It produces the product Q' * Z, where Z is an input
   * vector and Q is the orthogonal matrix diagonalizing the input matrix.
   *
   * @param d The diagonal elements of the input matrix. On output, d is overwritten by the eigenvalues of the symmetric
   * tridiagonal matrix.
   * @param e The subdiagonal elements of the input matrix. On output, the information in e has been overwritten. Has to
   * be of size d.size(), though the last element is irrelevant.
   * @param z On input, a vector. On output, the value of Q' * Z, where Q is the matrix that diagonalizes the input
   * symmetric tridiagonal matrix. Note that the columns of Q are the eigenvectors of the input matrix, and Q' is the
   * transpose of Q.
   */
  template <typename T>
  void diagonalize_tridiagonal_symmetric_matrix(std::vector<T> &d, std::vector<T> &e, std::vector<T> &z)
  {
    const int n = d.size();
    if (n == 1) return;
    if ((int)e.size() != n) throw std::runtime_error("The size of e must be d.size() for the purpose of this routine.");
    if ((int)z.size() != n) throw std::runtime_error("The size of z must be d.size().");
    T b = 0;
    T c = 0;
    T f = 0;
    T g = 0;
    int i = 0;
    int ii = 0;
    int itn = 30;
    int j = 0;
    int k = 0;
    int l = 0;
    int m = 0;
    int mml = 0;
    T p = 0;
    T prec = 0;
    T r = 0;
    T s = 0;

    prec = std::numeric_limits<T>::epsilon();

    using std::fabs, DiFfRG::sign;

    e[n - 1] = 0.0;

    for (l = 1; l <= n; ++l) {
      j = 0;
      for (;;) {
        for (m = l; m <= n; ++m) {
          if (m == n) break;
          if (fabs(e[m - 1]) <= prec * (fabs(d[m - 1]) + fabs(d[m]))) break;
        }

        p = d[l - 1];
        if (m == l) break;

        if (itn <= j) {
          std::cerr << "\n";
          std::cerr << "IMTQLX - Fatal error!\n";
          std::cerr << "  Iteration limit exceeded\n";
          throw std::runtime_error("IMTQLX - Fatal error!");
        }

        j = j + 1;
        g = (d[l] - p) / (2.0 * e[l - 1]);
        r = sqrt(g * g + 1.0);
        g = d[m - 1] - p + e[l - 1] / (g + fabs(r) * sign(g));
        s = 1.0;
        c = 1.0;
        p = 0.0;
        mml = m - l;

        for (ii = 1; ii <= mml; ++ii) {
          i = m - ii;
          f = s * e[i - 1];
          b = c * e[i - 1];

          if (fabs(g) <= fabs(f)) {
            c = g / f;
            r = sqrt(c * c + 1.0);
            e[i] = f * r;
            s = 1.0 / r;
            c = c * s;
          } else {
            s = f / g;
            r = sqrt(s * s + 1.0);
            e[i] = g * r;
            c = 1.0 / r;
            s = s * c;
          }
          g = d[i] - p;
          r = (d[i - 1] - g) * s + 2.0 * c * b;
          p = s * r;
          d[i] = g + p;
          g = c * r - b;
          f = z[i];
          z[i] = s * z[i - 1] + c * f;
          z[i - 1] = c * z[i - 1] - s * f;
        }
        d[l - 1] = d[l - 1] - p;
        e[l - 1] = g;
        e[m - 1] = 0.0;
      }
    }

    //  Sorting.
    for (ii = 2; ii <= m; ++ii) {
      i = ii - 1;
      k = i;
      p = d[i - 1];

      for (j = ii; j <= n; ++j) {
        if (d[j - 1] < p) {
          k = j;
          p = d[j - 1];
        }
      }

      if (k != i) {
        d[k - 1] = d[i - 1];
        d[i - 1] = p;
        p = z[i - 1];
        z[i - 1] = z[k - 1];
        z[k - 1] = p;
      }
    }
  }
} // namespace DiFfRG