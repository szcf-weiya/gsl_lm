#include <iostream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
using namespace std;

// Xv contain intercept
logit::logit(const gsl_vector *yv, const gsl_matrix *Xv)
{
  free_data = false;

  n = yv->size;
  p = Xv->size2;

  y = yv;
  X = Xv;

  // initialize
  beta = gsl_vector_calloc(p);
  gsl_vector_set_zero(beta);
}

// Xv does NOT contain the intercept
logit::logit(const std::vector<double> & yv, const std::vector<std::vector<double> > Xv)
{
  free_data = true; // need to free data

  n = yv.size();
  p = 1 + Xv.size();

  y = gsl_vector_calloc(n);
  X = gsl_matrix_calloc(n, p);
  for (size_t i = 0; i < n; i++)
  {
    gsl_vector_set(y, i, yv[i]);
    gsl_matrix_set(X, i, 0, 1.0); // intercept
    for (size_t j = 0; j < p - 1; j++)
    {
      gsl_matrix_set(X, i, j, Xv[j][i]);
    }
  }
}

~logit()
{

}

double logit::calculate_pi(gsl_vector* xi) const
{
  gsl_vector_mul(xi, beta);
  return(exp(xi)/(1+exp(xi)))
}

int logit::calculate_J(gsl_matrix* J) const
{
  double res, pii;
  gsl_vector *xj, *xk, *pi_1_pi, *xi;
  xj = gsl_vector_calloc(n);
  xk = gsl_vector_calloc(n);
  pi_1_pi = gsl_vector_calloc(n); // pi*(1-pi)
  xi = gsl_vector_calloc(p);
  for (size_t i = 0; i < n; i++)
  {
    gsl_matrix_get_row(xi, X, i);
    pii = calculate_pi(xi);
    gsl_vector_set(pi_1_pi, i, pii*(1-pii));
  }
  for (size_t j = 0; j < p; j++)
  {
    for (size_t k = 0; k < p; k++)
    {
      res = 0;
      gsl_matrix_get_col(xj, X, j);
      gsl_matrix_get_col(xk, X, k);
      // calculate
      gsl_vector_mul(pi_1_pi, xj);
      gsl_vector_mul(pi_1_pi, xk);
      for (size_t i = 0; i < n; i++)
        res += gsl_vector_get(pi_1_pi, i);
      gsl_matrix_set(J, j, k, res);
    }
  }

  gsl_vector_free(xi);
  gsl_vector_free(xj);
  gsl_vector_free(xk);
  gsl_vector_free(pi_1_pi);
  return 1;
}

int logit::calculate_U(gsl_vector* U) const
{
  double res;
  gsl_vector *xj, *y_pi, *pi, *xi;
  xj = gsl_vector_calloc(n);
  y_pi = gsl_vector_calloc(n); // y-pi
  xi = gsl_vector_calloc(p);
  gsl_vector_memcpy(y_pi, y);
  for (size_t i = 0; i < n; i++)
  {
    gsl_matrix_get_row(X, i, xi);
    gsl_vector_set(pi, i, calculate_pi(xi));
  }

  for (size_t j = 0; j < p; j++)
  {
    res = 0;
    gsl_matrix_get_col(xj, X, j);
    gsl_vector_sub(y_pi, pi);
    gsl_vector_mul(y_pi, xj);
    for (size_t i = 0; i < n; i++)
    {
      res += gsl_vector_get(y_pi, i);
    }
    gsl_vector_set(U, j, res);
  }
  gsl_vector_free(xi);
  gsl_vector_free(xj);
  gsl_vector_free(y_pi);
  gsl_vector_free(pi);
  return 1;
}

void logit::fit()
{
  gsl_matrix *J;
  J = gsl_matrix_calloc(p, p);
  gsl_vector *U, *beta2;
  U = gsl_vector_calloc(p);
  beta2 = gsl_vector_calloc(p);
  gsl_permutation *p = gsl_permutation_alloc(p);
  int s;
  int iter = 0;
  double err;
  while(true)
  {
      // solve beta
      err = 0
      gsl_linalg_LU_decomp(J, p, &s);
      gsl_linalg_LU_solve(J, p, Jbeta, beta2);
      err = calculate_err(beta2);
      cout << "iter = " << iter << " ... beta = " << beta
           <<" err = " << err << endl;
      if (err < ERR)
      {
        cout << "Finish!!" << endl;
        break;
      }

      // update J
      calculate_J(J);
      // update U
      calculate_U(U);
      // update Jbeta
      Jbeta -= U;
      iter++;
  }
}

double logit::calculate_err(const gsl_vector* beta2) const
{
    gsl_vector *beta3 = gsl_vector_alloc(p);
    gsl_vector_memcpy(beta3, beta2);
    gsl_vector_sub(beta3, beta);
    double err = 0, tmp;
    for (size_t i = 0; i < p; i++)
    {
      tmp = gsl_vector_get(beta3, i);
      err += tmp^2;
    }
    gsl_vector_free(beta3);
    return err;
}
