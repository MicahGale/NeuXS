/* Copyright (c) 2012 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* Available at: http://ab-initio.mit.edu/Faddeeva

   Header file for Faddeeva.cc; see that file for more information. */

#ifndef FADDEEVA_HH
#define FADDEEVA_HH 1

#include <thrust/complex.h>

namespace Faddeeva {

// compute w(z) = exp(-z^2) erfc(-iz) [ Faddeeva / scaled complex error func ]
template <typename FPrecision>
__host__ __device__ thrust::complex<FPrecision> w(thrust::complex<FPrecision> z,
                                                  FPrecision relerr = 0);

// special-case Im[w(x)] for real x
template <typename FPrecision>
__host__ __device__ FPrecision w_im(FPrecision x);

// compute erfcx(x) = exp(x^2) erfc(x) for real x (used internally by erfc)
template <typename FPrecision>
__host__ __device__ FPrecision erfcx(FPrecision x);

// compute erfc(z) = 1 - erf(z), the complementary error function
template <typename FPrecision>
__host__ __device__ thrust::complex<FPrecision>
erfc(thrust::complex<FPrecision> z, FPrecision relerr = 0);

} // namespace Faddeeva

#endif // FADDEEVA_HH
