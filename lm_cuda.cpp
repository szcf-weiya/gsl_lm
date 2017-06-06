#include<iostream>
#include<iomanip>
#include<fstream>
#include<sstream>
#include<vector>
#include<string>
#include<stdlib.h>
#include<stdio.h>
#include<time.h>
clock_t start, stop;
#include<gsl/gsl_vector.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_combination.h>
#include<gsl/gsl_statistics.h>
#include<gsl/gsl_fit.h>
#include<gsl/gsl_multifit.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_cdf.h>
using namespace std;

#define coef(i) (gsl_vector_get(coef, (i)))
#define COV(i, j) (gsl_matrix_get(cov, (i), (j)))

int lm(gsl_vector* y, size_t nrow_y, gsl_matrix* x, size_t nrow_x, size_t ncol_x, size_t flag)
{
  flag = 1; // exist intercept
  if (nrow_y != nrow_x)
    {
      cout << "The dimensions don't match." << endl;
      return 0;
    }
  size_t df;
  double tss, R2, adjR2, F, mpvalue;
  if (ncol_x == 1)
    {
      double c0, c1, cov00, cov01, cov11, sumsq;
      gsl_vector *xx;
      xx = gsl_vector_alloc(nrow_x);
      gsl_matrix_get_col(xx, x, 0);
      gsl_fit_linear(xx->data, 1, y->data, 1, nrow_y, &c0, &c1, &cov00, &cov01, &cov11, &sumsq);
      cout << "Coefficients\tEstimate\tStd. Error\tt value\tPr(>|t|)" << endl;
      // for intercept
      double stderr0 = sqrt(cov00);
      double t0 = c0/stderr0;
      double pvalue0 = 2*(t0 < 0 ? (1 - gsl_cdf_tdist_P(-t0, nrow_y - 2)) : (1 - gsl_cdf_tdist_P(t0, nrow_y - 2)));
      cout << "Intercept\t" << c0 << "\t" << stderr0 << "\t" << t0 << "\t" << pvalue0 << endl;

      // for coef
      double stderr1 = sqrt(cov11);
      double t1 = c1/stderr1;
      double pvalue1 = 2*(t1 < 0 ? (1 - gsl_cdf_tdist_P(-t1, nrow_y - 2)) : (1 - gsl_cdf_tdist_P(t1, nrow_y - 2)));
      cout << "x\t" << c1 << "\t" << stderr1 << "\t" << t1 << "\t" << pvalue1 << endl;

      df = nrow_y - 2;
      double tss = 0;
      double mu = gsl_stats_mean(y->data, 1, nrow_y);
      gsl_vector_add_constant(y, -mu);
      /*
      for (size_t i = 0; i < nrow_y; i++)
	{
	  tss += gsl_vector_get(y, i) *  gsl_vector_get(y, i);
	}
      */
      tss = gsl_stats_tss(y->data, 1, nrow_y);
      R2 = 1 - sumsq/tss;
      adjR2 = 1 - (nrow_y-1)/df*(1-R2);
      cout << "Multiple R-squared: " << R2 << ",\tAdjusted R-squared: " << adjR2 << endl;

      F = (tss - sumsq)/(sumsq/(nrow_y-2));
      mpvalue = 1 - gsl_cdf_fdist_P(F, 1, df);
      cout << "F-statistic: " << F << " on 1 and " << nrow_y - 2 << " DF, p-value: " << mpvalue << endl;
    }
  else
    {
      gsl_matrix *X;
      gsl_vector *tmp;
      X = gsl_matrix_alloc(nrow_x, ncol_x + 1);
      tmp = gsl_vector_alloc(nrow_x);
      for (size_t i = 0; i < ncol_x; i++)
	{
	  gsl_matrix_get_col(tmp, x, i);
	  gsl_matrix_set_col(X, i+1, tmp);
	}
      gsl_vector_set_all(tmp, 1);
      gsl_matrix_set_col(X, 0, tmp);
      
      gsl_matrix *cov;
      gsl_vector *coef;
      double chisq;
      cov = gsl_matrix_alloc(ncol_x + 1, ncol_x + 1);
      coef = gsl_vector_alloc(ncol_x + 1);
      
      gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(nrow_y, ncol_x + 1);
      gsl_multifit_linear(X, y, coef, cov, &chisq, work);
      gsl_multifit_linear_free(work);
      double stderr[ncol_x+1], pvalue[ncol_x+1], t[ncol_x+1];
      df = nrow_y - ncol_x - 1;
      
      for (size_t i = 0; i < ncol_x+1; i++)
	{
	  stderr[i] = sqrt(gsl_matrix_get(cov, i, i));
	  t[i] = gsl_vector_get(coef, i)/stderr[i];
	  pvalue[i] = 2*(t[i] < 0 ? (1 - gsl_cdf_tdist_P(-t[i], df)) : (1 - gsl_cdf_tdist_P(t[i], df)));
	}
      
      tss = gsl_stats_tss(y->data, 1, nrow_y);
      R2 = 1 - chisq/tss;
      adjR2 = 1 - (nrow_x - 1)/df*(1 - R2);
      F = ((tss - chisq)/ncol_x)/(chisq/df);
      mpvalue = 1 - gsl_cdf_fdist_P(F, ncol_x, df);
      cout << "Coefficients\tEstimate\tStd. Error\tt value\tPr(>|t|)" << endl;
      cout << "Intercept\t" << coef(0) << "\t" << stderr[0] << "\t" << t[0] << "\t" << pvalue[0] << endl;
      for (size_t i = 1; i <= ncol_x; i++)
	cout << "x" << i <<"\t" << coef(i) << "\t" << stderr[i] << "\t" << t[i] << "\t" << pvalue[i] << endl;
      
      cout << "Multiple R-squared: " << R2 << ",\tAdjusted R-squared: " << adjR2 << endl;
      cout << "F-statistic: " << F << " on 1 and " << nrow_y - 2 << " DF, p-value: " << mpvalue << endl;
    }
  return 1;
}
int main()
{
  gsl_matrix *x;
  gsl_vector *y;
  x = gsl_matrix_alloc(4, 2);
  y = gsl_vector_alloc(4);
  const gsl_rng_type *T;
  gsl_rng *r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  double yy[4] = {1.1, 2, 2.9, 4.2};
  double xx[4] = {1, 2, 2, 1};
  for (size_t i = 0; i < 4; i++)
    {
      gsl_matrix_set(x, i, 1, xx[i]);
      gsl_matrix_set(x, i, 0, i);
      gsl_vector_set(y, i, yy[i]);
    }
  lm(y, 4, x, 4, 2, 1);
  
  return 0;
}

