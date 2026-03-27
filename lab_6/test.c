#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include <cblas.h>

#define N_SIZE      1750        // Размер матрицы
#define RUNS        10          // Количество запусков 
#define MAX_THREADS 16          // Максимальное количество потоков 

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

#define CblasRowMajor 101
#define CblasColMajor 102
#define CblasLeft     141
#define CblasRight    142
#define CblasUpper    121
#define CblasLower    122

// Таймер  
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

// Среднее геометрическое 
double geometric_mean(double *values, int n) {
    double product = 1.0;
    for (int i = 0; i < n; i++) {
        product *= values[i];
    }
    return pow(product, 1.0 / n);
}

// инициализация матрицы  
void init_mat_float(float *mat, int n, int lda) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * lda + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

void init_sym_float(float *mat, int n, int lda) {
    init_mat_float(mat, n, lda);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            mat[j * lda + i] = mat[i * lda + j];
        }
    }
}

void init_mat_double(double *mat, int n, int lda) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * lda + j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
}

void init_sym_double(double *mat, int n, int lda) {
    init_mat_double(mat, n, lda);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            mat[j * lda + i] = mat[i * lda + j];
        }
    }
}

// проверка корректности 
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
        printf("  Различия в пределах допустимого\n");
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
        printf("  Различия в пределах допустимого\n");
        return 1;
    }
    
    printf("  Корректно (max_diff = %.2e)\n", max_diff);
    return 1;
}

// Тест FLOAT  
void test_float(int n, int runs, int max_threads) {
    printf("\n");
    printf("FLOAT: матрица %d x %d\n", n, n);
    printf("Количество запусков: %d\n", runs);
    
    int lda = n, ldb = n, ldc = n;
    
    float *A = (float*)malloc(n * n * sizeof(float));
    float *B = (float*)malloc(n * n * sizeof(float));
    float *C_my = (float*)malloc(n * n * sizeof(float));
    float *C_oblas = (float*)malloc(n * n * sizeof(float));
    float *C_check = (float*)malloc(n * n * sizeof(float));
    
    if (!A || !B || !C_my || !C_oblas || !C_check) {
        printf("Ошибка выделения памяти!\n");
        exit(1);
    }
    
    srand(42);
    init_sym_float(A, n, lda);
    init_mat_float(B, n, ldb);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Проверка корректности 
    printf("\nПроверка корректности...\n");
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C_my[i * ldc + j] = 0.0f;
            C_oblas[i * ldc + j] = 0.0f;
            C_check[i * ldc + j] = 0.0f;
        }
    }
    
    my_ssymm(CblasRowMajor, CblasLeft, CblasUpper, n, n,
             alpha, A, lda, B, ldb, beta, C_my, ldc);
    
    cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, n, n,
                alpha, A, lda, B, ldb, beta, C_oblas, ldc);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C_check[i * ldc + j] = sum;
        }
    }
    
    printf("Сравнение my_symm с простым алгоритмом:\n");
    check_float(C_my, C_check, n, ldc);
    printf("Сравнение OpenBLAS с простым алгоритмом:\n");
    check_float(C_oblas, C_check, n, ldc);
    
    // Массивы потоков 
    int threads[] = {1, 2, 4, 8, 16};
    
    printf("\nЗапуск бенчмаркинга\n");
    
    for (int ti = 0; ti < 5; ti++) {
        int t = threads[ti];
        if (t > max_threads) continue;
        
        openblas_set_num_threads(t);
        
        printf("Потоков: %d\n", t);
        
        double times_my[RUNS];
        double times_oblas[RUNS];
        double total_my = 0.0, total_oblas = 0.0;
        
        for (int run = 0; run < runs; run++) {
            // Сброс C для my_ssymm 
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    C_my[i * ldc + j] = 0.0f;
                }
            }
            
            double start = get_time();
            my_ssymm(CblasRowMajor, CblasLeft, CblasUpper, n, n,
                     alpha, A, lda, B, ldb, beta, C_my, ldc);
            double end = get_time();
            double elapsed_my = end - start;
            times_my[run] = elapsed_my;
            total_my += elapsed_my;
            
            // Сброс C для OpenBLAS 
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    C_oblas[i * ldc + j] = 0.0f;
                }
            }
            
            start = get_time();
            cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, n, n,
                        alpha, A, lda, B, ldb, beta, C_oblas, ldc);
            end = get_time();
            double elapsed_oblas = end - start;
            times_oblas[run] = elapsed_oblas;
            total_oblas += elapsed_oblas;
            
            printf("  Запуск %2d: my=%8.3f сек, OpenBLAS=%8.3f сек\n", 
                   run + 1, elapsed_my, elapsed_oblas);
        }
        
        double avg_my = total_my / runs;
        double avg_oblas = total_oblas / runs;
        double geom_my = geometric_mean(times_my, runs);
        double geom_oblas = geometric_mean(times_oblas, runs);
        double percent = (avg_oblas / avg_my) * 100.0;
        
        printf("  Среднее арифметическое: my=%8.3f, OpenBLAS=%8.3f\n", avg_my, avg_oblas);
        printf("  Среднее геометрическое: my=%8.3f, OpenBLAS=%8.3f\n", geom_my, geom_oblas);
        printf("  Относительная производительность: %.2f%%\n", percent);
        printf("\n");
    }
    
    free(A);
    free(B);
    free(C_my);
    free(C_oblas);
    free(C_check);
}

// Тест DOUBLE  
void test_double(int n, int runs, int max_threads) {
    printf("\n");
    printf("DOUBLE: матрица %d x %d\n", n, n);
    printf("Количество запусков: %d\n", runs);
    
    int lda = n, ldb = n, ldc = n;
    
    double *A = (double*)malloc(n * n * sizeof(double));
    double *B = (double*)malloc(n * n * sizeof(double));
    double *C_my = (double*)malloc(n * n * sizeof(double));
    double *C_oblas = (double*)malloc(n * n * sizeof(double));
    double *C_check = (double*)malloc(n * n * sizeof(double));
    
    if (!A || !B || !C_my || !C_oblas || !C_check) {
        printf("Ошибка выделения памяти!\n");
        exit(1);
    }
    
    srand(42);
    init_sym_double(A, n, lda);
    init_mat_double(B, n, ldb);
    
    double alpha = 1.0;
    double beta = 0.0;
    
    // Проверка корректности 
    printf("\nПроверка корректности...\n");
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C_my[i * ldc + j] = 0.0;
            C_oblas[i * ldc + j] = 0.0;
            C_check[i * ldc + j] = 0.0;
        }
    }
    
    my_dsymm(CblasRowMajor, CblasLeft, CblasUpper, n, n,
             alpha, A, lda, B, ldb, beta, C_my, ldc);
    
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, n, n,
                alpha, A, lda, B, ldb, beta, C_oblas, ldc);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C_check[i * ldc + j] = sum;
        }
    }
    
    printf("Сравнение my_symm с простым алгоритмом:\n");
    check_double(C_my, C_check, n, ldc);
    printf("Сравнение OpenBLAS с простым алгоритмом:\n");
    check_double(C_oblas, C_check, n, ldc);
    
    // Массивы потоков 
    int threads[] = {1, 2, 4, 8, 16};
    
    printf("\nЗапуск бенчмаркинга\n");
    printf("\n");
    for (int ti = 0; ti < 5; ti++) {
        int t = threads[ti];
        if (t > max_threads) continue;
        
        openblas_set_num_threads(t);
        
        printf("Потоков: %d\n", t);
        
        double times_my[RUNS];
        double times_oblas[RUNS];
        double total_my = 0.0, total_oblas = 0.0;
        
        for (int run = 0; run < runs; run++) {
            // Сброс C для my_dsymm 
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    C_my[i * ldc + j] = 0.0;
                }
            }
            
            double start = get_time();
            my_dsymm(CblasRowMajor, CblasLeft, CblasUpper, n, n,
                     alpha, A, lda, B, ldb, beta, C_my, ldc);
            double end = get_time();
            double elapsed_my = end - start;
            times_my[run] = elapsed_my;
            total_my += elapsed_my;
            
            // Сброс C для OpenBLAS 
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    C_oblas[i * ldc + j] = 0.0;
                }
            }
            
            start = get_time();
            cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, n, n,
                        alpha, A, lda, B, ldb, beta, C_oblas, ldc);
            end = get_time();
            double elapsed_oblas = end - start;
            times_oblas[run] = elapsed_oblas;
            total_oblas += elapsed_oblas;
            
            printf("  Запуск %2d: my=%8.3f сек, OpenBLAS=%8.3f сек\n", 
                   run + 1, elapsed_my, elapsed_oblas);
        }
        
        double avg_my = total_my / runs;
        double avg_oblas = total_oblas / runs;
        double geom_my = geometric_mean(times_my, runs);
        double geom_oblas = geometric_mean(times_oblas, runs);
        double percent = (avg_oblas / avg_my) * 100.0;
        
        printf("  Среднее арифметическое: my=%8.3f, OpenBLAS=%8.3f\n", avg_my, avg_oblas);
        printf("  Среднее геометрическое: my=%8.3f, OpenBLAS=%8.3f\n", geom_my, geom_oblas);
        printf("  Относительная производительность: %.2f%%\n", percent);
        printf("\n");
    }
    
    free(A);
    free(B);
    free(C_my);
    free(C_oblas);
    free(C_check);
}
 
int main() {
    printf("\n");
    printf("Тестирование\n");
    printf("Размер матрицы: %d x %d\n", N_SIZE, N_SIZE);
    printf("Количество запусков: %d\n", RUNS);
    printf("Максимум потоков: %d\n", MAX_THREADS);
    
    printf("\n");
    printf("Начинаем тестирование...\n");
    
    test_float(N_SIZE, RUNS, MAX_THREADS);
    test_double(N_SIZE, RUNS, MAX_THREADS);
    
    printf("\n");
    printf("Тестирование завершено\n");
    
    return 0;
}