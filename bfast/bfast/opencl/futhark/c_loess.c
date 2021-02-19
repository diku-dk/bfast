#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct LOESS_Results {
  double * result;
  double * slopes;
} LOESS_Result;


double * new_arr(int length) {
  double * ans = (double *) calloc(length, sizeof(double));
  return ans;
}


LOESS_Result c_loess(
  double* xx,       // time values - should be 1:n unless there are some NAs
  double* yy,       // the corresponding y values
  int degree,       // degree of smoothing
  int span,         // span of smoothing
  double * ww,      // weights
  int* m,           // points at which to evaluate the smooth
  int* l_idx,       // index of left starting points
  double* max_dist, // distance between nn bounds for each point
  int n,
  int n_m
) {
  int span2, span3, offset;
  int i, j;
  double r, tmp1, tmp2;

  double* x = new_arr(span);
  double* w = new_arr(span);
  double* xw = new_arr(span);
  double* x2w = new_arr(span);

  double* result = new_arr(n_m);
  double* slopes = new_arr(n_m);

  // variables for storing determinant intermediate values
  double a, b, c, d, e, a1, b1, c1, a2, b2, c2, det;

  if(span > n) {
    span = n;
  }

  // want to start storing results at index 0, corresponding to the lowest m
  offset = m[0];

  // loop through all values of m
  for(i = 0; i < n_m; i++) {
    a = 0.0;

    // get weights, x, and a
    for(j = 0; j < span; j++) {
      w[j] = 0.0;
      x[j] = xx[l_idx[i] + j] - (double)m[i];

      /* r = std::fabs(x[j]); */
      r = fabs(x[j]);
      // tricube

      tmp1 = r / max_dist[i];
      // manual multiplication is much faster than pow()
      tmp2 = 1.0 - tmp1 * tmp1 * tmp1;
      w[j] = tmp2 * tmp2 * tmp2;

      // scale by user-defined weights
      w[j] = w[j] * ww[l_idx[i] + j];

      a = a + w[j];
    }

    if(degree == 0) {
       a1 = 1 / a;
       for(j = 0; j < span; j++) {
          result[i] = result[i] + w[j] * a1 * yy[l_idx[i] + j];
       }
    } else {
      // get xw, x2w, b, c for degree 1 or 2
      b = 0.0;
      c = 0.0;
      for(j = 0; j < span; j++) {
        xw[j] = x[j] * w[j];
        x2w[j] = x[j] * xw[j];
        b = b + xw[j];
        c = c + x2w[j];
      }
      if(degree == 1) {
        det = 1 / (a * c - b * b);
        a1 = c * det;
        b1 = -b * det;
        c1 = a * det;
        for(j=0; j < span; j++) {
          result[i] += (w[j] * a1 + xw[j] * b1) * yy[l_idx[i] + j];
          slopes[i] += (w[j] * b1 + xw[j] * c1) * yy[l_idx[i] + j];
        }
      }
    }
  }

  free(x);
  free(w);
  free(xw);
  free(x2w);

  LOESS_Result res;
  res.result = result;

  res.slopes = slopes;

  return res;
}


int main()
{
  double xx[] = {1, 2, 3, 4, 5};             // time values - should be 1:n unless there are some NA
  double yy[] = {1.2, 2.3, 3.2, 4.1, 5.3};   // the corresponding y value
  int degree = 1;                            // degree of smoothing
  int span = 2;                              // span of smoothing
  double ww[] = {1, 1, 1, 1, 1};             // weights
  /* int m[] = {5, 7};                          // points at which to evaluate the smooth */
  int m[] = {2, 3, 4};                          // points at which to evaluate the smooth
  /* int l_idx[] = {1, 2};                      // index of left starting points */
  int l_idx[] = {0, 1, 2};                      // index of left starting points
  /* double max_dist[] = {3.43, 5.32};          // distance between nn bounds for each point */
  double max_dist[] = {3.43, 5.32, 6.1};          // distance between nn bounds for each point
  int n = 5;
  /* int n_m = 2;                               // length of m */
  int n_m = 3;                               // length of m

  LOESS_Result res = c_loess(xx, yy, degree, span, ww, m, l_idx, max_dist, n, n_m);
  double * result = res.result;
  double * slopes = res.slopes;

  printf("result: ");
  for (int i = 0; i < n_m; ++i) {
    printf("%f, ", result[i]);
  }
  printf("\nslopes: ");
  for (int i = 0; i < n_m; ++i) {
    printf("%f, ", slopes[i]);
  }

  free(result);
  free(slopes);
  return 0;
}
