#include <stdio.h>
#include <stdlib.h>


double* c_interp(double* m, double* fits, double* slopes, double* at, int n_at) {
  int i, j;
  double u, h, u2, u3;

  double* ans = (double *) malloc (sizeof(double) * n_at);

  j = 0; // index of leftmost vertex

  int* indxs = (int *) malloc (sizeof(int) * n_at);

  for(i = 0; i < n_at; ++i) {
    if(at[i] > m[j + 1]) {
      j++;
    }
    printf("i = %i, j = %i\n", i, j);
    h = (m[j + 1] - m[j]);
    u = (at[i] -  m[j]) / h;
    u2 = u * u;
    u3 = u2 * u;
    ans[i] = (2 * u3 - 3 * u2 + 1) * fits[j] +
             (3 * u2 - 2 * u3)     * fits[j + 1] +
             (u3 - 2 * u2 + u)     * slopes[j] * h +
             (u3 - u2)             * slopes[j + 1] * h;
  }

  free(indxs);
  return ans;
}


int main() {
  /* 3 5 */
  double m[] = {1, 3, 5, 7};
  double fits[] = {1.1, 3.4, 5.3, 7.8};
  double slopes[] = {0, 0, 0, 0};
  double at[] = {1, 2, 3, 4, 5, 6, 7};
  int n_at = 7;

  /* double m[] = {5, 7}; */
  /* double fits[] = {5.3, 7.8}; */
  /* double slopes[] = {0, 0}; */
  /* double at[] = {1, 2, 3, 4, 5, 6, 7}; */
  /* int n_at = 7; */

  /* 5 */
  /* double m[] = {3, 5, 7}; */
  /* double fits[] = {3.4, 5.3, 7.8}; */
  /* double slopes[] = {0, 0, 0}; */
  /* double at[] = {1, 2, 3, 4, 5, 6, 7}; */
  /* int n_at = 7; */

  /* 6 */
  /* double m[] = {5, 6, 7}; */
  /* double fits[] = {5.4, 6.1, 7.8}; */
  /* double slopes[] = {0, 0, 0}; */
  /* double at[] = {1, 2, 3, 4, 5, 6, 7}; */
  /* int n_at = 7; */

  /* 5 6 */
  /* double m[] = {4, 5, 6, 7}; */
  /* double fits[] = {4.1, 5.4, 6.1, 7.8}; */
  /* double slopes[] = {0, 0, 0, 0}; */
  /* double at[] = {1, 2, 3, 4, 5, 6, 7}; */
  /* int n_at = 7; */

  /* 4, 5, 6 */
  /* double m[] = {3, 4, 5, 6, 7}; */
  /* double fits[] = {3.3, 4.1, 5.4, 6.1, 7.8}; */
  /* double slopes[] = {0, 0, 0, 0, 0}; */
  /* double at[] = {1, 2, 3, 4, 5, 6, 7}; */
  /* int n_at = 7; */

  /* /\* 4*\/ */
  /* double m[] = {1, 4, 7}; */
  /* double fits[] = {1.3, 4.8, 7.8}; */
  /* double slopes[] = {0, 0, 0}; */
  /* double at[] = {1, 2, 3, 4, 5, 6, 7}; */
  /* int n_at = 7; */


  double * out = c_interp(m, fits, slopes, at, n_at);
  for (int i = 0; i < n_at; ++i) {
    printf("%f; ", out[i]);
  }
  free(out);
  return 0;
}

/*** R
     if(any(is.nan(fits))) {
     ind <- !is.nan(fits)
     c_interp(m[ind], fits[ind], slopes[ind], at)
     } else {
     c_interp(m, fits, slopes, at)
     }
*/
