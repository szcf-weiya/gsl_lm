#include <iostream>
using namespace std;
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_cdf.h>

#include "logit.h"

void display(gsl_matrix* m)
{
  for (size_t i = 0; i < m->size1; i++)
  {
    for (size_t j = 0; j < m->size2; j++)
    {
      cout << gsl_matrix_get(m, i, j) << " ";
    }
    cout << endl;
  }
  cout << endl;
}


int main()
{

  // load response vector

  size_t N = 601;
  gsl_vector * y = gsl_vector_alloc(N);
  FILE * f = fopen("ynaffair.dat", "r");
  gsl_vector_fread(f, y);
  fclose(f);

  // load predictor matrix
  size_t P = 4+1;
  gsl_matrix * X = gsl_matrix_alloc(N, P);
  f = fopen("x.dat", "r");
  gsl_matrix_fread(f, X);
  fclose(f);

  logit Affairs(y, X);

/*
  double y_data[] = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  double x_data[] = {1, 10,25,1, 5, 27, 1, 8, 18,1,  11, 11,1, 12, 28, 1,14, 3,1,3, 6, 1,1, 30,1, 16, 26, 1,18, 22};
  gsl_vector_view y = gsl_vector_view_array(y_data, 10);
  gsl_matrix_view X = gsl_matrix_view_array(x_data, 10, 3);

  logit Affairs(&y.vector, &X.matrix);
  */
  cout << "Hello" << endl;
  //gsl_vector_free(y);
  //gsl_matrix_free(X);
  return 0;
}
