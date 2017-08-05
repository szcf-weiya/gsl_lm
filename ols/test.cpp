#include <iostream>
//#include "cuLUsolve.h"
#include "cuMultifit.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

using namespace std;

void printMatrix(gsl_matrix *m)
{
  for (size_t i = 0; i < m->size1; i++)
  {
    for (size_t j = 0; j < m->size2; j++)
    {
      cout << gsl_matrix_get(m, i, j) << ", ";
    }
    cout << endl;
  }
}

int main()
{
  const gsl_rng_type *T;
  gsl_rng *r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);

  // test inv(A)
  /*
  int N = 5;
  double *A = (double*)malloc(sizeof(double)*N*N);
  gsl_matrix *B = gsl_matrix_alloc(N, N);
  gsl_matrix_set_identity(B);

  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      A[i+j*N] = gsl_rng_uniform(r);

  gsl_matrix_view mA = gsl_matrix_view_array(A, N, N);

  printMatrix(&mA.matrix);

  cuda_LU_solve(A, N, B->data, N);
  printMatrix(B);
  */

  // test ols
  double A[] = {1, 1, 1, 1, 2, 3, 5, 4};
  double B[] = {1, 2, 3, 4};
  double coef[2];
  double pvalue[2];

  cuMultifit(A, 4, 2, B, coef, pvalue);

  cout << coef[0] << " " << pvalue[0] << endl
       << coef[1] << " " << pvalue[1] << endl;

  gsl_rng_free(r);
  return 0;
}
