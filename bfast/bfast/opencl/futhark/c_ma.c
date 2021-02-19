#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int print_array(double* arr, int n) {
  for (int i = 0; i < n; ++i) {
    printf("%f, ", arr[i]);
  }
  printf("\n");
}

double * new_arr(int length) {
  double * ans = (double *) calloc(length, sizeof(double));
  return ans;
}

/* Moving averages */
double* c_ma(double* x, int n, int n_p) {
  int i;
  int nn = n - n_p * 2;
  double ma_tmp = 0;
  double ma_tmp1 = 0;
  double ma_tmp2 = 0;

  double* ans = new_arr(n - 2 * n_p);
  double* ma  = new_arr(nn + n_p + 1);
  double* ma2 = new_arr(nn + 2);
  double* ma3 = new_arr(nn);

  ma_tmp = 0;
  for(i = 0; i < n_p; ++i) {
    ma_tmp = ma_tmp + x[i];
  }
  printf("%f\n", ma_tmp);
  ma[0] = ma_tmp / n_p;

  for(i = 0; i < nn + n_p; ++i) {
    ma_tmp += x[i + n_p] - x[i];
    ma[i + 1] = ma_tmp / n_p;
  }

  for(i = 0; i < n_p; ++i) {
    ma_tmp1 = ma_tmp1 + ma[i];
  }
  ma2[0] = ma_tmp1 / n_p;

  for(i = 0; i < nn + 1; ++i) {
    ma_tmp1 = ma_tmp1 + ma[i + n_p] - ma[i];
    ma2[i + 1] = ma_tmp1 / n_p;
  }

  for(i = 0; i < 3; ++i) {
    ma_tmp2 = ma_tmp2 + ma2[i];
  }
  ans[0] = ma_tmp2 / 3;

  for(i = 0; i < nn - 1; ++i) {
    ma_tmp2 = ma_tmp2 - ma2[i] + ma2[i + 3];
    ans[i + 1] = ma_tmp2 / 3;
  }

  return ans;
}


int main()
{
  double x[] = {1.2, 2.3, 3.2, 4.1, 5.3, 6.1, 7.3, 8.1, 9.3, 10.12};
  int n = 10;
  int n_p = 2;
  int n_res = n - n_p * 2;

  double* res = c_ma(x, n, n_p);
  exit(0);
  for (int i = 0; i < n_res; ++i) {
    printf("%f, ", res[i]);
  }
  printf("\n");

  free(res);
  return 0;
}
