#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CBLAS_INDEX size_t
#include <stddef.h>  // для size_t

// ========== ПРАВИЛЬНЫЕ ИНТЕРФЕЙСЫ, НО НЕПРАВИЛЬНАЯ ЛОГИКА ==========

// 1. SAXPY - правильный интерфейс, но неправильная логика
void cblas_saxpy(const int n, const float alpha, const float *x, const int incx, float *y, const int incy) {
    printf(" SAXPY: возвращаю неправильный результат (все нули)\n");
    for (int i = 0; i < n; i++) {
        y[i * incy] = 0.0f;  // Должно быть y = alpha*x + y
    }
}

// 2. SCOPY - правильный интерфейс, но копирует неправильно
void cblas_scopy(const int n, const float *x, const int incx, float *y, const int incy) {
    printf(" SCOPY: копирую только первый элемент\n");
    if (n > 0) {
        y[0] = x[0];  // Копирует только первый элемент
    }
}

// 3. SDOT - правильный интерфейс, но возвращает неправильное значение
float cblas_sdot(const int n, const float *x, const int incx, const float *y, const int incy) {
    printf(" SDOT: возвращаю 0.0 всегда\n");
    return 0.0f;  // Должно быть сумма произведений
}

// 4. SNRM2 - правильный интерфейс, но возвращает 1.0 всегда
float cblas_snrm2(const int n, const float *x, const int incx) {
    printf(" SNRM2: возвращаю 1.0 всегда\n");
    return 1.0f;  // Должна быть норма
}

// 5. ISAMAX - правильный интерфейс, но возвращает 0 всегда
CBLAS_INDEX cblas_isamax(const int n, const float *x, const int incx) {
    printf(" ISAMAX: возвращаю 0 всегда\n");
    return 0;  // Должен быть индекс максимума
}

// 6. SSWAP - правильный интерфейс, но не меняет местами
void cblas_sswap(const int n, float *x, const int incx, float *y, const int incy) {
    printf(" SSWAP: ничего не меняю\n");
    // Ничего не делает - должна быть замена
}

// 7. DAXPY - правильный интерфейс, но неправильная логика
void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy) {
    printf(" DAXPY: добавляю только alpha, без x\n");
    for (int i = 0; i < n; i++) {
        y[i * incy] = alpha;  // Должно быть alpha*x + y
    }
}

// 8. DCOPY - правильный интерфейс, но копирует со смещением
void cblas_dcopy(const int n, const double *x, const int incx, double *y, const int incy) {
    printf(" DCOPY: копирую со сдвигом на 1\n");
    for (int i = 0; i < n - 1; i++) {
        y[(i + 1) * incy] = x[i * incx];  // Копирует со сдвигом
    }
}

// 9. DDOT - правильный интерфейс, но возвращает неправильное значение
double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy) {
    printf(" DDOT: возвращаю 42.0 всегда\n");
    return 42.0;  // Всегда возвращает 42
}

// 10. DNRM2 - правильный интерфейс, но возвращает 2.0 всегда
double cblas_dnrm2(const int n, const double *x, const int incx) {
    printf(" DNRM2: возвращаю 2.0 всегда\n");
    return 2.0;  // Всегда возвращает 2
}

// 11. IDAMAX - правильный интерфейс, но возвращает последний индекс
CBLAS_INDEX cblas_idamax(const int n, const double *x, const int incx) {
    printf(" IDAMAX: возвращаю последний индекс\n");
    return n - 1;  // Возвращает последний, а не максимальный
}

// 12. DSWAP - правильный интерфейс, но меняет неправильно
void cblas_dswap(const int n, double *x, const int incx, double *y, const int incy) {
    printf(" DSWAP: обмениваю только первый элемент\n");
    if (n > 0) {
        double temp = x[0];
        x[0] = y[0];
        y[0] = temp;
    }
}