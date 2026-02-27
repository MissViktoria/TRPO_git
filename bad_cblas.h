#ifndef BAD_CBLAS_H
#define BAD_CBLAS_H

#include <stddef.h>  // для size_t

#define CBLAS_INDEX size_t

// Функции float (s)
void cblas_saxpy(const int n, const float alpha, const float *x, const int incx, float *y, const int incy);
void cblas_scopy(const int n, const float *x, const int incx, float *y, const int incy);
float cblas_sdot(const int n, const float *x, const int incx, const float *y, const int incy);
float cblas_snrm2(const int n, const float *x, const int incx);
CBLAS_INDEX cblas_isamax(const int n, const float *x, const int incx);
void cblas_sswap(const int n, float *x, const int incx, float *y, const int incy);

// Функции double (d)
void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy);
void cblas_dcopy(const int n, const double *x, const int incx, double *y, const int incy);
double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy);
double cblas_dnrm2(const int n, const double *x, const int incx);
CBLAS_INDEX cblas_idamax(const int n, const double *x, const int incx);
void cblas_dswap(const int n, double *x, const int incx, double *y, const int incy);

#endif