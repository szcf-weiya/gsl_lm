#include <iostream>
#include <iomanip>
#include "logit.h"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cdf.h>

using namespace std;

#define ERR 1e-10

// Xv contain intercept
logit::logit(gsl_vector *yv, gsl_matrix *Xv)
{
  free_data = false;

  n = yv->size;
  p = Xv->size2;

  y = yv;
  X = Xv;

  // initialize
  beta = gsl_vector_calloc(p);
  Jbeta = gsl_vector_calloc(p);
  pvalue = gsl_vector_calloc(p);
//  gsl_vector_set_zero(beta);
  gsl_vector_set_all(beta, 0);
  /*
  gsl_vector_set(beta, 0, -0.1);
  gsl_vector_set(beta, 1, 0.7);
  gsl_vector_set(beta, 2, 1);
  */
  /*
  gsl_vector_set(beta, 1, 1.5);
  gsl_vector_set(beta, 4, -0.5);
  double a[] = {1, 2, 3, 4, 5};
  gsl_vector_view bb = gsl_vector_view_array(a, 5);
  calculate_pi(&bb.vector);
  */
  fit();
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

logit::~logit()
{
  if (free_data)
  {
    gsl_vector_free(y);
    gsl_matrix_free(X);
    gsl_vector_free(beta);
    gsl_vector_free(Jbeta);
    gsl_vector_free(pvalue);
  }
}

double logit::calculate_pi(gsl_vector* xi) const
{
  gsl_vector_mul(xi, beta);
  double res = 0;

  for (size_t i = 0; i < p; i++)
    res += gsl_vector_get(xi, i);
  return(1.0/(1+exp(-1.0*res)));
}

int logit::calculate_J(gsl_matrix* J) const
{
  double res, pii;
  gsl_vector *xj, *xk, *pi_1_pi, *xi, *tmp;
  xj = gsl_vector_calloc(n);
  xk = gsl_vector_calloc(n);
  pi_1_pi = gsl_vector_calloc(n); // pi*(1-pi)
  tmp = gsl_vector_calloc(n);
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
      // reset tmp to record results
      gsl_vector_memcpy(tmp, pi_1_pi);

      gsl_matrix_get_col(xj, X, j);
      gsl_matrix_get_col(xk, X, k);
      // calculate
      gsl_vector_mul(tmp, xj);
      gsl_vector_mul(tmp, xk);
      for (size_t i = 0; i < n; i++)
        res += gsl_vector_get(tmp, i);
      gsl_matrix_set(J, j, k, -1.0*res);
    }
  }

  gsl_vector_free(xi);
  gsl_vector_free(xj);
  gsl_vector_free(xk);
  gsl_vector_free(pi_1_pi);
  gsl_vector_free(tmp);
  return 1;
}

int logit::calculate_U(gsl_vector* U) const
{
  double res;
  gsl_vector *xj, *y_pi, *pi, *xi, *tmp;
  xj = gsl_vector_calloc(n);
  y_pi = gsl_vector_calloc(n); // y-pi
  tmp = gsl_vector_calloc(n);
  pi = gsl_vector_calloc(n);
  xi = gsl_vector_calloc(p);

  gsl_vector_memcpy(y_pi, y);
  for (size_t i = 0; i < n; i++)
  {
    gsl_matrix_get_row(xi, X, i);
    gsl_vector_set(pi, i, calculate_pi(xi));
  }
  gsl_vector_sub(y_pi, pi);

  for (size_t j = 0; j < p; j++)
  {
    res = 0;
    // reset
    gsl_vector_memcpy(tmp, y_pi);

    gsl_matrix_get_col(xj, X, j);
    gsl_vector_mul(tmp, xj);
    for (size_t i = 0; i < n; i++)
    {
      res += gsl_vector_get(tmp, i);
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
  gsl_matrix *J, *Jinv;
  J = gsl_matrix_calloc(p, p);
  Jinv = gsl_matrix_calloc(p, p);
  gsl_vector_view Jinvdiag;
  gsl_vector *U, *beta2, *tau;
  U = gsl_vector_calloc(p);
  beta2 = gsl_vector_calloc(p);
  tau = gsl_vector_alloc(p);
  gsl_permutation *permulation = gsl_permutation_alloc(p);
  int s;
  int iter = 0;
  double err, err_min = 1e40;
  // initialize
  //cout << "Before ................\n";
  //cout << "beta = " << endl;
  //displayv(beta);
  //cout << "J =  " << endl;
  //display(J);
  calculate_J(J);
  //cout << "beta = " << endl;
  //displayv(beta);
  //cout << "J =  " << endl;
  display(J);
  //cout << "After initialize, J =  "<< endl;
  calculate_U(U);
  //cout << "U =  " << endl;
  //displayv(U);
  // Jbeta
  gsl_blas_dgemv(CblasNoTrans, 1.0, J, beta, 0, beta2);
  gsl_vector_memcpy(Jbeta, beta2);
  while(true)
  {
    // update Jbeta
    gsl_vector_sub(Jbeta, U);
    // solve beta
    err = 0;
    // LU
    display(J);
    gsl_linalg_LU_decomp(J, permulation, &s);
    gsl_linalg_LU_solve(J, permulation, Jbeta, beta2);

    // QR
    /*
    gsl_linalg_QR_decomp(J, tau);
    gsl_linalg_QR_solve(J, tau, Jbeta, beta2);
    */
    err = calculate_err(beta2);
    /*
    if (err < err_min)
      err_min = err;
    else
    {
      cout << "Finish!!" << endl;
      break;
    }
    */

    gsl_vector_memcpy(beta, beta2);
    cout << "iter = " << iter
         <<" err = " << err
         << " beta = ";
    for (size_t i = 0; i < p; i++)
      cout << gsl_vector_get(beta,  i) << " ";
    cout << endl;
    if (err < ERR)
    {
      cout << "Finish!!" << endl;
      break;
    }
    if (iter > 10)
      break;
    // update J
    calculate_J(J);
    // update U
    calculate_U(U);
    // update Jbeta
    gsl_blas_dgemv(CblasNoTrans, 1.0, J, beta, 0, beta2);
    gsl_vector_memcpy(Jbeta, beta2);
    iter++;
  }
  // make sure J is LU decomposition
  // inverse (for p-value)
  gsl_linalg_LU_invert(J, permulation, Jinv);
  Jinvdiag = gsl_matrix_subdiagonal(Jinv, 0);
  double zscore, betai, se, pvaluei;
  for (size_t i = 0; i < p; i++)
  {
    betai = gsl_vector_get(beta,  i);
    se = sqrt(-1.0*gsl_vector_get(&Jinvdiag.vector, i));
    zscore = betai/se;
    pvaluei = 2*(zscore < 0 ? gsl_cdf_gaussian_P(zscore, 1) : gsl_cdf_gaussian_P(-zscore, 1));
    cout << "i = " << i << " zscore = " << zscore << " p-value = " << pvaluei << endl;
  }
  cout << endl;
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
      err += tmp * tmp;
    }
    gsl_vector_free(beta3);
    return err;
}

double logit::calculate_err() const
{
    gsl_vector *tmp = gsl_vector_alloc(n);
    gsl_vector *xi = gsl_vector_alloc(p);
    gsl_vector *pihat = gsl_vector_alloc(n);
    gsl_vector_memcpy(tmp, y);
    for (size_t i = 0; i < n; i++)
    {
      gsl_matrix_get_row(xi, X, i);
      gsl_vector_set(pihat, i,  calculate_pi(xi));
    }
    gsl_vector_sub(tmp, pihat);
    double res = 0;
    double s;
    for (size_t i = 0; i < n; i++)
    {
      s = gsl_vector_get(tmp, i);
      res += s*s;
    }
    return res;
}

void logit::display(gsl_matrix* m) const
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
void logit::displayv(gsl_vector* v) const
{
  for (size_t i = 0; i < v->size; i++)
  {
    cout << gsl_vector_get(v, i) << " ";
  }
  cout << endl;
}
