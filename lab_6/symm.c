#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };

// float 

// Вспомогательная функция для Left side (C = alpha*A*B + beta*C) 
static void my_ssymm_left(int M, int N, int uplo, float alpha,
                          const float *A, int lda,
                          const float *B, int ldb, float beta,
                          float *C, int ldc) {
    int i, j, k;
    
    // Умножаем C на beta 
    if (beta != 1.0f) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                if (beta == 0.0f) {
                    C[i * ldc + j] = 0.0f;
                } else {
                    C[i * ldc + j] *= beta;
                }
            }
        }
    }
    
    // Добавляем alpha * A * B 
    if (alpha != 0.0f) {
        if (uplo == CblasUpper) {
            // Используем верхний треугольник матрицы A 
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    float sum = 0.0f;
                    //    A[k][i] для k <= i
                    //    A[i][k] для k > i 
                    for (k = 0; k <= i; k++) {
                        sum += A[k * lda + i] * B[k * ldb + j];
                    }
                    for (k = i + 1; k < M; k++) {
                        sum += A[i * lda + k] * B[k * ldb + j];
                    }
                    C[i * ldc + j] += alpha * sum;
                }
            }
        } else { // CblasLower 
            // Используем нижний треугольник матрицы A 
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    float sum = 0.0f;
                    //    A[i][k] для k < i
                    //    A[k][i] для k >= i 
                    for (k = 0; k < i; k++) {
                        sum += A[i * lda + k] * B[k * ldb + j];
                    }
                    for (k = i; k < M; k++) {
                        sum += A[k * lda + i] * B[k * ldb + j];
                    }
                    C[i * ldc + j] += alpha * sum;
                }
            }
        }
    }
}

// Вспомогательная функция для Right side (C = alpha*B*A + beta*C) 
static void my_ssymm_right(int M, int N, int uplo, float alpha,
                           const float *A, int lda,
                           const float *B, int ldb, float beta,
                           float *C, int ldc) {
    int i, j, k;
    
    // Умножаем C на beta 
    if (beta != 1.0f) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                if (beta == 0.0f) {
                    C[i * ldc + j] = 0.0f;
                } else {
                    C[i * ldc + j] *= beta;
                }
            }
        }
    }
    
    // Добавляем alpha * B * A 
    if (alpha != 0.0f) {
        if (uplo == CblasUpper) {
            // Используем верхний треугольник матрицы A (размер N x N) 
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (k = 0; k <= j; k++) {
                        sum += B[i * ldb + k] * A[k * lda + j];
                    }
                    for (k = j + 1; k < N; k++) {
                        sum += B[i * ldb + k] * A[j * lda + k];
                    }
                    C[i * ldc + j] += alpha * sum;
                }
            }
        } else { // CblasLower 
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (k = 0; k < j; k++) {
                        sum += B[i * ldb + k] * A[j * lda + k];
                    }
                    for (k = j; k < N; k++) {
                        sum += B[i * ldb + k] * A[k * lda + j];
                    }
                    C[i * ldc + j] += alpha * sum;
                }
            }
        }
    }
}

// Основная функция symm для float 
void my_ssymm(int order, int side, int uplo,
              int M, int N,
              float alpha, const float *A, int lda,
              const float *B, int ldb, float beta,
              float *C, int ldc) {
    
    // Проверяем поддерживаемый формат 
    if (order != CblasRowMajor) {
        fprintf(stderr, "Ошибка: поддерживается только CblasRowMajor\n");
        return;
    }
    
    if (side == CblasLeft) {
        my_ssymm_left(M, N, uplo, alpha, A, lda, B, ldb, beta, C, ldc);
    } else { 
        //CblasRight
        my_ssymm_right(M, N, uplo, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

// double

static void my_dsymm_left(int M, int N, int uplo, double alpha,
                          const double *A, int lda,
                          const double *B, int ldb, double beta,
                          double *C, int ldc) {
    int i, j, k;
    
    if (beta != 1.0) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                if (beta == 0.0) {
                    C[i * ldc + j] = 0.0;
                } else {
                    C[i * ldc + j] *= beta;
                }
            }
        }
    }
    
    if (alpha != 0.0) {
        if (uplo == CblasUpper) {
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    double sum = 0.0;
                    for (k = 0; k <= i; k++) {
                        sum += A[k * lda + i] * B[k * ldb + j];
                    }
                    for (k = i + 1; k < M; k++) {
                        sum += A[i * lda + k] * B[k * ldb + j];
                    }
                    C[i * ldc + j] += alpha * sum;
                }
            }
        } else {
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    double sum = 0.0;
                    for (k = 0; k < i; k++) {
                        sum += A[i * lda + k] * B[k * ldb + j];
                    }
                    for (k = i; k < M; k++) {
                        sum += A[k * lda + i] * B[k * ldb + j];
                    }
                    C[i * ldc + j] += alpha * sum;
                }
            }
        }
    }
}

static void my_dsymm_right(int M, int N, int uplo, double alpha,
                           const double *A, int lda,
                           const double *B, int ldb, double beta,
                           double *C, int ldc) {
    int i, j, k;
    
    if (beta != 1.0) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                if (beta == 0.0) {
                    C[i * ldc + j] = 0.0;
                } else {
                    C[i * ldc + j] *= beta;
                }
            }
        }
    }
    
    if (alpha != 0.0) {
        if (uplo == CblasUpper) {
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    double sum = 0.0;
                    for (k = 0; k <= j; k++) {
                        sum += B[i * ldb + k] * A[k * lda + j];
                    }
                    for (k = j + 1; k < N; k++) {
                        sum += B[i * ldb + k] * A[j * lda + k];
                    }
                    C[i * ldc + j] += alpha * sum;
                }
            }
        } else {
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    double sum = 0.0;
                    for (k = 0; k < j; k++) {
                        sum += B[i * ldb + k] * A[j * lda + k];
                    }
                    for (k = j; k < N; k++) {
                        sum += B[i * ldb + k] * A[k * lda + j];
                    }
                    C[i * ldc + j] += alpha * sum;
                }
            }
        }
    }
}

void my_dsymm(int order, int side, int uplo,
              int M, int N,
              double alpha, const double *A, int lda,
              const double *B, int ldb, double beta,
              double *C, int ldc) {
    
    if (order != CblasRowMajor) {
        fprintf(stderr, "Ошибка: поддерживается только CblasRowMajor\n");
        return;
    }
    
    if (side == CblasLeft) {
        my_dsymm_left(M, N, uplo, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        my_dsymm_right(M, N, uplo, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}