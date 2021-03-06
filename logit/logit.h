#ifndef _LOGIT_H
#define _LOGIT_H


#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>


class logit
{
  private:
    gsl_vector *y; // response vector
    gsl_matrix *X; // observation matrix

    size_t n; // sample size;
    size_t p; // number of parameters (including intercept)

    double psi; // dispersion

    bool free_data; // depend on different constructors

    gsl_vector *beta;
    gsl_vector *Jbeta;
    gsl_vector *pvalue;

  public:
    logit(gsl_vector *yv, gsl_matrix *Xv);
    logit(const std::vector<double> & yv, const std::vector<std::vector<double> > Xv);
    ~logit();
    void fit();
    double calculate_pi(gsl_vector* xi) const;
    double calculate_err(const gsl_vector* beta2) const;
    double calculate_err() const;
    int calculate_J(gsl_matrix* J) const;
    int calculate_U(gsl_vector* U) const;
    void display(gsl_matrix* m) const;
    void displayv(gsl_vector* v) const;
}; //logit

#endif // _LOGIT_H
