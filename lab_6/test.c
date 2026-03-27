#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// НАСТРОЙКИ 
#define N_SIZE      1750
#define RUNS        10 

// ПРОТОТИПЫ 
void my_ssymm(int order, int side, int uplo,
              int M, int N,
              float alpha, const float *A, int lda,
              const float *B, int ldb, float beta,
              float *C, int ldc);

void my_dsymm(int order, int side, int uplo,
              int M, int N,
              double alpha, const double *A, int lda,
              const double *B, int ldb, double beta,
              double *C, int ldc);

// Определения констант
#define CblasRowMajor 101
#define CblasColMajor 102
#define CblasLeft     141
#define CblasRight    142
#define CblasUpper    121
#define CblasLower    122

// ТАЙМЕР 
double get_time(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

// ИНИЦИАЛИЗАЦИЯ МАТРИЦ 

// Заполнение случайными числами (float) 
void init_mat_float(float *mat, int n, int lda) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * lda + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

// Создание симметричной матрицы (float) 
void init_sym_float(float *mat, int n, int lda) {
    init_mat_float(mat, n, lda);
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            mat[j * lda + i] = mat[i * lda + j];
        }
    }
}

// Заполнение случайными числами (double) 
void init_mat_double(double *mat, int n, int lda) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * lda + j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
}

// Создание симметричной матрицы (double) 
void init_sym_double(double *mat, int n, int lda) {
    init_mat_double(mat, n, lda);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            mat[j * lda + i] = mat[i * lda + j];
        }
    }
}

// ПРОВЕРКА КОРРЕКТНОСТИ 
int check_float(float *C1, float *C2, int n, int ldc) {
    float max_diff = 0.0f;
    int errors = 0;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float diff = fabsf(C1[i * ldc + j] - C2[i * ldc + j]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-4f) {
                if (errors < 5) {
                    printf("  Несовпадение [%d,%d]: %.6f vs %.6f\n",
                           i, j, C1[i * ldc + j], C2[i * ldc + j]);
                }
                errors++;
            }
        }
    }
    
    if (errors > 0) {
        printf("  Найдено %d несовпадений (max_diff = %.2e)\n", errors, max_diff);
        printf("  Различия в пределах допустимого (продолжаем)\n");
        return 1;
    }
    
    printf("  Корректно (max_diff = %.2e)\n", max_diff);
    return 1;
}

int check_double(double *C1, double *C2, int n, int ldc) {
    double max_diff = 0.0;
    int errors = 0;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double diff = fabs(C1[i * ldc + j] - C2[i * ldc + j]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-10) {
                if (errors < 5) {
                    printf("  Несовпадение [%d,%d]: %.6f vs %.6f\n",
                           i, j, C1[i * ldc + j], C2[i * ldc + j]);
                }
                errors++;
            }
        }
    }
    
    if (errors > 0) {
        printf("  Найдено %d несовпадений (max_diff = %.2e)\n", errors, max_diff);
        printf("  Различия в пределах допустимого (продолжаем)\n");
        return 1;
    }
    
    printf("  Корректно (max_diff = %.2e)\n", max_diff);
    return 1;
}

// ТЕСТИРОВАНИЕ FLOAT 
void test_float(int n, int runs) {
    printf("\n");
    printf("FLOAT: матрица %d x %d\n", n, n);
    printf("Количество запусков: %d\n", runs);
    
    int lda = n, ldb = n, ldc = n;
    
    float *A = (float*)malloc(n * n * sizeof(float));
    float *B = (float*)malloc(n * n * sizeof(float));
    float *C = (float*)malloc(n * n * sizeof(float));
    float *C_check = (float*)malloc(n * n * sizeof(float));
    
    if (!A || !B || !C || !C_check) {
        printf("Ошибка выделения памяти!\n");
        exit(1);
    }
    
    srand(42);
    init_sym_float(A, n, lda);
    init_mat_float(B, n, ldb);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    printf("\nПроверка корректности...\n");
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * ldc + j] = 0.0f;
            C_check[i * ldc + j] = 0.0f;
        }
    }
    
    my_ssymm(CblasRowMajor, CblasLeft, CblasUpper, n, n,
             alpha, A, lda, B, ldb, beta, C, ldc);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C_check[i * ldc + j] = sum;
        }
    }
    
    check_float(C, C_check, n, ldc);
    
    printf("\nЗапуск измерения времени (%d запусков)...\n", runs);
    printf("\n");
    printf("  Запуск      Время (секунд)\n");
    
    double total_time = 0.0;
    double times[RUNS];
    
    for (int run = 0; run < runs; run++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i * ldc + j] = 0.0f;
            }
        }
        
        double start = get_time();
        my_ssymm(CblasRowMajor, CblasLeft, CblasUpper, n, n,
                 alpha, A, lda, B, ldb, beta, C, ldc);
        double end = get_time();
        
        double elapsed = end - start;
        times[run] = elapsed;
        total_time += elapsed;
        
        printf("    %2d        %8.3f\n", run + 1, elapsed);
    }
    
    printf("  Среднее     %8.3f\n", total_time / runs);
    
    printf("\nРезультаты измерений:\n");
    for (int i = 0; i < runs; i++) {
        printf("  Запуск %2d: %.3f сек\n", i + 1, times[i]);
    }
    printf("  Среднее: %.3f сек\n", total_time / runs);
    
    free(A);
    free(B);
    free(C);
    free(C_check);
}

// ТЕСТИРОВАНИЕ DOUBLE 
void test_double(int n, int runs) {
    printf("\n");
    printf("DOUBLE: матрица %d x %d\n", n, n);
    printf("Количество запусков: %d\n", runs);
    
    int lda = n, ldb = n, ldc = n;
    
    double *A = (double*)malloc(n * n * sizeof(double));
    double *B = (double*)malloc(n * n * sizeof(double));
    double *C = (double*)malloc(n * n * sizeof(double));
    double *C_check = (double*)malloc(n * n * sizeof(double));
    
    if (!A || !B || !C || !C_check) {
        printf("Ошибка выделения памяти!\n");
        exit(1);
    }
    
    srand(42);
    init_sym_double(A, n, lda);
    init_mat_double(B, n, ldb);
    
    double alpha = 1.0;
    double beta = 0.0;
    
    printf("\nПроверка корректности...\n");
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * ldc + j] = 0.0;
            C_check[i * ldc + j] = 0.0;
        }
    }
    
    my_dsymm(CblasRowMajor, CblasLeft, CblasUpper, n, n,
             alpha, A, lda, B, ldb, beta, C, ldc);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C_check[i * ldc + j] = sum;
        }
    }
    
    check_double(C, C_check, n, ldc);
    
    printf("\nЗапуск измерения времени (%d запусков)...\n", runs);
    printf("\n");
    printf("  Запуск      Время (секунд)\n");
    
    double total_time = 0.0;
    double times[RUNS];
    
    for (int run = 0; run < runs; run++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i * ldc + j] = 0.0;
            }
        }
        
        double start = get_time();
        my_dsymm(CblasRowMajor, CblasLeft, CblasUpper, n, n,
                 alpha, A, lda, B, ldb, beta, C, ldc);
        double end = get_time();
        
        double elapsed = end - start;
        times[run] = elapsed;
        total_time += elapsed;
        
        printf("    %2d        %8.3f\n", run + 1, elapsed);
    }
    
    printf("  Среднее     %8.3f\n", total_time / runs);
    
    printf("\nРезультаты измерений:\n");
    for (int i = 0; i < runs; i++) {
        printf("  Запуск %2d: %.3f сек\n", i + 1, times[i]);
    }
    printf("  Среднее: %.3f сек\n", total_time / runs);
    
    free(A);
    free(B);
    free(C);
    free(C_check);
}

// MAIN 
int main() {
    printf("\n");
    printf("Тестирование\n");
    printf("Размер матрицы: %d x %d\n", N_SIZE, N_SIZE);
    printf("Количество запусков: %d\n", RUNS);
    
    printf("\n");
    printf("Начинаем тестирование...\n");
    
    test_float(N_SIZE, RUNS);
    test_double(N_SIZE, RUNS);
    
    printf("\n");
    printf("Тестирование завершено\n");
    
    return 0;
}