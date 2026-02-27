#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cblas.h"

#define MAX_N 10
#define EPSILON 1e-6

typedef struct {
    char test_name[50];
    int passed;
    char message[100];
} TestResult;

TestResult results[100];
int test_count = 0;

int assert_float_eq(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

int assert_double_eq(double a, double b, double eps) {
    return fabs(a - b) < eps;
}

void record_result(const char* name, int passed, const char* message) {
    strcpy(results[test_count].test_name, name);
    results[test_count].passed = passed;
    strcpy(results[test_count].message, message);
    test_count++;
    printf("%s: %s\n", name, passed ? "PASS" : "FAIL");
    if (!passed) printf("  Message: %s\n", message);
}

// ==================== FLOAT (s) ====================

void test_scopy_float(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float expected[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int n = 5;
    
    cblas_scopy(n, x, 1, y, 1);
    
    int passed = 1;
    for (int i = 0; i < n; i++) {
        if (!assert_float_eq(y[i], expected[i], EPSILON)) passed = 0;
    }
    record_result("scopy (float)", passed, "Копирование вектора float");
}

void test_saxpy_float(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {1.0f, 1.0f, 1.0f};
    float expected[] = {3.5f, 6.0f, 8.5f};
    int n = 3;
    float alpha = 2.5f;
    
    cblas_saxpy(n, alpha, x, 1, y, 1);
    
    int passed = 1;
    for (int i = 0; i < n; i++) {
        if (!assert_float_eq(y[i], expected[i], EPSILON)) passed = 0;
    }
    record_result("saxpy (float)", passed, "y = 2.5*x + y");
}

void test_sdot_float(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {4.0f, 5.0f, 6.0f};
    float expected = 32.0f;
    int n = 3;
    
    float result = cblas_sdot(n, x, 1, y, 1);
    
    int passed = assert_float_eq(result, expected, EPSILON);
    char msg[100];
    sprintf(msg, "Ожидалось: %f, Получено: %f", expected, result);
    record_result("sdot (float)", passed, msg);
}

void test_snrm2_float(void) {
    float x[] = {3.0f, 4.0f};
    float expected = 5.0f;
    int n = 2;
    
    float result = cblas_snrm2(n, x, 1);
    
    int passed = assert_float_eq(result, expected, EPSILON);
    char msg[100];
    sprintf(msg, "Ожидалось: %f, Получено: %f", expected, result);
    record_result("snrm2 (float)", passed, msg);
}

void test_isamax_float(void) {
    float x[] = {-5.0f, 1.0f, 3.0f, -2.0f, 4.0f};
    int expected = 0;
    int n = 5;
    
    int result = cblas_isamax(n, x, 1);
    
    int passed = (result == expected);
    char msg[100];
    sprintf(msg, "Ожидалось: %d, Получено: %d", expected, result);
    record_result("isamax (float)", passed, msg);
}

void test_sswap_float(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {4.0f, 5.0f, 6.0f};
    float expected_x[] = {4.0f, 5.0f, 6.0f};
    float expected_y[] = {1.0f, 2.0f, 3.0f};
    int n = 3;
    
    cblas_sswap(n, x, 1, y, 1);
    
    int passed = 1;
    for (int i = 0; i < n; i++) {
        if (!assert_float_eq(x[i], expected_x[i], EPSILON) ||
            !assert_float_eq(y[i], expected_y[i], EPSILON)) passed = 0;
    }
    record_result("sswap (float)", passed, "Обмен векторов");
}

// ==================== DOUBLE (d) ====================

void test_dcopy_double(void) {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double expected[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int n = 5;
    
    cblas_dcopy(n, x, 1, y, 1);
    
    int passed = 1;
    for (int i = 0; i < n; i++) {
        if (!assert_double_eq(y[i], expected[i], EPSILON)) passed = 0;
    }
    record_result("dcopy (double)", passed, "Копирование вектора double");
}

void test_daxpy_double(void) {
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {1.0, 1.0, 1.0};
    double expected[] = {3.5, 6.0, 8.5};
    int n = 3;
    double alpha = 2.5;
    
    cblas_daxpy(n, alpha, x, 1, y, 1);
    
    int passed = 1;
    for (int i = 0; i < n; i++) {
        if (!assert_double_eq(y[i], expected[i], EPSILON)) passed = 0;
    }
    record_result("daxpy (double)", passed, "y = 2.5*x + y");
}

void test_ddot_double(void) {
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    double expected = 32.0;
    int n = 3;
    
    double result = cblas_ddot(n, x, 1, y, 1);
    
    int passed = assert_double_eq(result, expected, EPSILON);
    char msg[100];
    sprintf(msg, "Ожидалось: %f, Получено: %f", expected, result);
    record_result("ddot (double)", passed, msg);
}

void test_dnrm2_double(void) {
    double x[] = {3.0, 4.0};
    double expected = 5.0;
    int n = 2;
    
    double result = cblas_dnrm2(n, x, 1);
    
    int passed = assert_double_eq(result, expected, EPSILON);
    char msg[100];
    sprintf(msg, "Ожидалось: %f, Получено: %f", expected, result);
    record_result("dnrm2 (double)", passed, msg);
}

void test_idamax_double(void) {
    double x[] = {-5.0, 1.0, 3.0, -2.0, 4.0};
    int expected = 0;
    int n = 5;
    
    int result = cblas_idamax(n, x, 1);
    
    int passed = (result == expected);
    char msg[100];
    sprintf(msg, "Ожидалось: %d, Получено: %d", expected, result);
    record_result("idamax (double)", passed, msg);
}

void test_dswap_double(void) {
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    double expected_x[] = {4.0, 5.0, 6.0};
    double expected_y[] = {1.0, 2.0, 3.0};
    int n = 3;
    
    cblas_dswap(n, x, 1, y, 1);
    
    int passed = 1;
    for (int i = 0; i < n; i++) {
        if (!assert_double_eq(x[i], expected_x[i], EPSILON) ||
            !assert_double_eq(y[i], expected_y[i], EPSILON)) passed = 0;
    }
    record_result("dswap (double)", passed, "Обмен векторов double");
}

// ==================== MAIN ====================

int main(void) {
    printf("\n");
    printf("----------------------------------\n");
    printf("CBLAS Level 1 Interface Tests\n");
    printf("----------------------------------\n\n");
    
    // Информация о потоках
    char* threads = getenv("OPENBLAS_NUM_THREADS");
    if (threads) {
        printf("OPENBLAS_NUM_THREADS = %s\n", threads);
    } else {
        printf("OPENBLAS_NUM_THREADS = 1 (по умолчанию)\n");
    }
    
    printf("\n=== FLOAT (s) ===\n");
    test_scopy_float();
    test_saxpy_float();
    test_sdot_float();
    test_snrm2_float();
    test_isamax_float();
    test_sswap_float();
    
    printf("\n=== DOUBLE (d) ===\n");
    test_dcopy_double();
    test_daxpy_double();
    test_ddot_double();
    test_dnrm2_double();
    test_idamax_double();
    test_dswap_double();
    
    printf("\n");
    printf("----------------------------------\n");
    printf("Итоги тестирования\n");
    printf("----------------------------------\n");
    
    int total_passed = 0;
    for (int i = 0; i < test_count; i++) {
        printf("%-20s: %s\n", results[i].test_name, 
               results[i].passed ? "PASS" : "FAIL");
        if (results[i].passed) total_passed++;
    }
    
    printf("----------------------------------\n");
    printf("Пройдено: %d/%d тестов (%.1f%%)\n", 
           total_passed, test_count, 
           test_count > 0 ? 100.0 * total_passed / test_count : 0);
    printf("----------------------------------\n");
    
    return (total_passed == test_count) ? 0 : 1;
}