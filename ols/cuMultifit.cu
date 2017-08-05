#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuLUsolve.h"
#include "cuMultifit.h"

int cuMultifit(const double *X, int n, int p, const double *Y, double *coef, double *cov)
{
  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;


  const int lda = n;
  double *C;
  C = (double*)malloc(sizeof(double)*p*p);

  double *d_X = NULL;
  double *d_C = NULL;
  double *d_Y = NULL;

  // create cublas handle
  cublas_status = cublasCreate(&cublasH);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);

  // copy to device
  cudaStat1 = cudaMalloc ((void**)&d_X, sizeof(double) * lda * p);
  cudaStat2 = cudaMalloc ((void**)&d_C, sizeof(double) * p * p);
  cudaStat3 = cudaMalloc ((void**)&d_Y, sizeof(double) * p);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3); //check!!

  cudaStat1 = cudaMemcpy(d_X, X, sizeof(double) * lda * p, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat1);
  cudaStat2 = cudaMemcpy(d_Y, Y, sizeof(double) * p, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat2);
  double alpha_v = 1.0;
  double beta_v = 0.0;
  const double *alpha = &alpha_v, *beta = &beta_v; //check!!
  printf("hello\n");
  printf("%f\n", *alpha);
  // d_C = d_X^T d_X
  cublas_status = cublasDgemm(cublasH,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           p, n, p,
                           alpha,
                           d_X, n,
                           d_X, n,
                           beta,
                           d_C, p);
  cudaStat1 = cudaDeviceSynchronize();
  assert(cublas_status == CUBLAS_STATUS_SUCCESS);
  assert(cudaSuccess == cudaStat1);
  printf("finish X'X\n");
  // copy d_C to C
  cudaStat1 = cudaMemcpy(C, d_C, sizeof(double)*p*p, cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat1);

  // inv(C)
  gsl_matrix *B = gsl_matrix_alloc(p, p);
  gsl_matrix_set_identity(B);
  cuda_LU_solve(C, p, B->data, p);
  cudaStat1 = cudaMemcpy(d_C, B->data, sizeof(double)*p*p, cudaMemcpyHostToDevice);
  gsl_matrix_free(B);
  assert(cudaSuccess == cudaStat1);
  printf("finish inv(C)\n");
  // d_Y = d_X^T * d_Y
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_T,
                           n, p,
                           alpha,
                           d_X, n,
                           d_Y, 1,
                           beta,
                           d_Y, 1);
  cudaStat1 = cudaDeviceSynchronize();
  assert(cublas_status == CUBLAS_STATUS_SUCCESS);
  assert(cudaSuccess == cudaStat1);

  // inv(C) * d_Y
  // due to by-column in gpu while by-row in gsl, C need to be transpose
  cublas_status = cublasDgemv(cublasH, CUBLAS_OP_T,
                           p, p,
                           alpha,
                           d_C, p,
                           d_Y, 1,
                           beta,
                           d_Y, 1);
  cudaStat1 = cudaDeviceSynchronize();
  assert(cublas_status == CUBLAS_STATUS_SUCCESS);
  assert(cudaSuccess == cudaStat1);

  // copy to coef
  cudaStat1 = cudaMemcpy(d_Y, coef, sizeof(double) * p, cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat1);

  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_C);

  cublasDestroy(cublasH);
  cudaDeviceReset();
  return 0;
}
