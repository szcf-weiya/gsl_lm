#include <cstdlib>
#include <cstring>
#include <cmath>

#include <vector>
#include <string>
#include <iostream>
using namespace std;

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_cdf.h>

#include "IRLS.h"

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

  
  // load offset vector
  gsl_vector * offset = NULL;
    // fit the model
  IRLS irls("log-link");
  irls.link->quasi = false;
  irls.set_data(y, X, offset);
  irls.fit_model();
  vector<double> coev = irls.get_coef();
  vector<double> sev = irls.get_stderr();
  
  // print the results
  printf("dispersion=%.4f\n", irls.get_dispersion());
  printf("%10s%12s%12s%15s\n", "", "Estimate", "Std.Error", "p-value");
  for(size_t i = 0; i < coev.size(); ++i){
    printf("X%-9zu%12.9f%12.8f", i, coev[i], sev[i]);
    if(! irls.link->quasi)
      printf("%15.6e\n", 2 * gsl_cdf_gaussian_P(-fabs(coev[i]/sev[i]), 1.0));
    else
      printf("%15.6e\n", 2 * gsl_cdf_tdist_P(-fabs(coev[i]/sev[i]), N-irls.get_rank_X()));
  }
  
  gsl_vector_free(y);
  gsl_matrix_free(X);
  gsl_vector_free(offset);
  
  return EXIT_SUCCESS;
}
