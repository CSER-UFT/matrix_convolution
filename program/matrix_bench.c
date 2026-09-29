#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "input_format.h"

#define CONV_LINEAR 1
#define CONV_MALLOC 2
#define GEMM_BLOCKED 3
#define FULL 0
#define APPROXIMATE 1

#ifndef OPERATION
#error OPERATION deve ser CONV_LINEAR, CONV_MALLOC ou GEMM_BLOCKED
#endif
#ifndef TECHNIQUE
#error TECHNIQUE deve ser FULL ou APPROXIMATE
#endif
#ifndef PROGRAM_NAME
#define PROGRAM_NAME "matrix_bench"
#endif

#ifdef USE_FLOAT
typedef float real_t;
#define PRECISION_NAME "float"
#define IS_FLOAT 1
#define INPUT_DATA_TYPE INPUT_TYPE_FLOAT
#else
typedef double real_t;
#define PRECISION_NAME "double"
#define IS_FLOAT 0
#define INPUT_DATA_TYPE INPUT_TYPE_DOUBLE
#endif

#if TECHNIQUE == APPROXIMATE && OPERATION == GEMM_BLOCKED
#define TECHNIQUE_NAME "skip_k"
#elif TECHNIQUE == APPROXIMATE
#define TECHNIQUE_NAME "skip_kernel"
#else
#define TECHNIQUE_NAME "full"
#endif

static int parse_positive(const char *text, int *value) {
    char *end = NULL;
    long parsed;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno || end == text || *end || parsed <= 0 || parsed > INT_MAX) return 0;
    *value = (int)parsed;
    return 1;
}

static real_t *copy_values(const real_t *source, uint64_t count) {
    real_t *result = malloc((size_t)count * sizeof(real_t));
    if (!result) return NULL;
    memcpy(result, source, (size_t)count * sizeof(real_t));
    return result;
}

#if OPERATION == CONV_LINEAR
static void operation(real_t *output, const real_t *matrix, int n,
                      const real_t *kernel, int k, real_t selected_sum) {
    int offset = k / 2;
    (void)selected_sum;
    for (int i = offset; i < n - offset; i++) {
        for (int j = offset; j < n - offset; j++) {
            real_t sum = (real_t)0;
#if TECHNIQUE == APPROXIMATE
            for (int ki = 0; ki < k; ki += 2)
                for (int kj = 0; kj < k; kj += 2)
                    sum += matrix[(i - offset + ki) * n + j - offset + kj] * kernel[ki * k + kj];
#else
            for (int ki = 0; ki < k; ki++)
                for (int kj = 0; kj < k; kj++)
                    sum += matrix[(i - offset + ki) * n + j - offset + kj] * kernel[ki * k + kj];
#endif
            output[i * n + j] = sum;
        }
    }
}
#elif OPERATION == CONV_MALLOC
static real_t **allocate_rows(int rows, int cols, int zero) {
    real_t **matrix = calloc((size_t)rows, sizeof(*matrix));
    if (!matrix) return NULL;
    for (int i = 0; i < rows; i++) {
        matrix[i] = zero ? calloc((size_t)cols, sizeof(real_t))
                         : malloc((size_t)cols * sizeof(real_t));
        if (!matrix[i]) {
            for (int j = 0; j < i; j++) free(matrix[j]);
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

static void release_rows(real_t **matrix, int rows) {
    if (!matrix) return;
    for (int i = 0; i < rows; i++) free(matrix[i]);
    free(matrix);
}

static void operation(real_t **output, real_t **matrix, int n,
                      real_t **kernel, int k, real_t selected_sum) {
    int offset = k / 2;
    (void)selected_sum;
    for (int i = offset; i < n - offset; i++) {
        for (int j = offset; j < n - offset; j++) {
            real_t sum = (real_t)0;
#if TECHNIQUE == APPROXIMATE
            for (int ki = 0; ki < k; ki += 2)
                for (int kj = 0; kj < k; kj += 2)
                    sum += matrix[i - offset + ki][j - offset + kj] * kernel[ki][kj];
#else
            for (int ki = 0; ki < k; ki++)
                for (int kj = 0; kj < k; kj++)
                    sum += matrix[i - offset + ki][j - offset + kj] * kernel[ki][kj];
#endif
            output[i][j] = sum;
        }
    }
}
#else
static void operation(real_t *output, const real_t *a, const real_t *b, int n, int block, real_t scale) {
#if TECHNIQUE == FULL
    (void)scale;
#endif
    for (int ii = 0; ii < n; ii += block)
        for (int jj = 0; jj < n; jj += block)
            for (int kk = 0; kk < n; kk += block)
                for (int i = ii; i < ii + block && i < n; i++)
                    for (int j = jj; j < jj + block && j < n; j++) {
                        real_t sum = output[i * n + j];
#if TECHNIQUE == APPROXIMATE
                        for (int k = kk + (kk & 1); k < kk + block && k < n; k += 2)
                            sum += a[i * n + k] * b[k * n + j] * scale;
#else
                        for (int k = kk; k < kk + block && k < n; k++)
                            sum += a[i * n + k] * b[k * n + j];
#endif
                        output[i * n + j] = sum;
                    }
}
#endif

int main(int argc, char **argv) {
    input_data_t input = {0}, kernel_input = {0};
    const char *dump_path = NULL;
    int parameter = 0, ok = 0;
    uint64_t matrix_count;
    real_t checksum = (real_t)0;

#if OPERATION == GEMM_BLOCKED
    if ((argc != 3 && argc != 5) || !parse_positive(argv[2], &parameter) ||
        (argc == 5 && strcmp(argv[3], "--dump") != 0)) {
        fprintf(stderr, "Uso: %s <entrada_gemm.bin> <bloco> [--dump arquivo]\n", argv[0]);
        return 1;
    }
    if (argc == 5) dump_path = argv[4];
    if (!input_load(argv[1], INPUT_GEMM, INPUT_DATA_TYPE, sizeof(real_t), &input)) return 1;
    matrix_count = (uint64_t)input.header.n * input.header.n;
    if (input.header.count != 2 * matrix_count) {
        fprintf(stderr, "Erro: tamanho incompativel na entrada GEMM.\n");
        goto cleanup;
    }
    const real_t *input_values = input.values;
    real_t *a = copy_values(input_values, matrix_count);
    real_t *b = copy_values(input_values + matrix_count, matrix_count);
    real_t *output = calloc((size_t)matrix_count, sizeof(real_t));
    real_t scale = (real_t)input.header.n / (real_t)((input.header.n + 1) / 2);
    if (!a || !b || !output) {
        fprintf(stderr, "Erro: falha de alocacao.\n");
        free(a); free(b); free(output); goto cleanup;
    }
    operation(output, a, b, (int)input.header.n, parameter, scale);
    for (uint64_t i = 0; i < matrix_count; i++) checksum += output[i];
    if (!isfinite(checksum)) { fprintf(stderr, "Erro: resultado nao finito.\n"); goto gemm_free; }
    if (dump_path && !dump_as_double(dump_path, output, matrix_count, IS_FLOAT)) {
        fprintf(stderr, "Erro: nao foi possivel gravar %s.\n", dump_path); goto gemm_free;
    }
    printf("Program=%s Operation=gemm Precision=%s Technique=%s N=%u Dist=-1 Parameter=%d Seed=%u\n",
           PROGRAM_NAME, PRECISION_NAME, TECHNIQUE_NAME, input.header.n, parameter, input.header.seed);
    printf("Checksum %.12f\n", (double)checksum);
    ok = 1;
gemm_free:
    free(a); free(b); free(output);
#else
    if ((argc != 3 && argc != 5) || (argc == 5 && strcmp(argv[3], "--dump") != 0)) {
        fprintf(stderr, "Uso: %s <matriz.bin> <kernel.bin> [--dump arquivo]\n", argv[0]);
        return 1;
    }
    if (argc == 5) dump_path = argv[4];
    if (!input_load(argv[1], INPUT_MATRIX, INPUT_DATA_TYPE, sizeof(real_t), &input) ||
        !input_load(argv[2], INPUT_KERNEL, INPUT_DATA_TYPE, sizeof(real_t), &kernel_input)) goto cleanup;
    matrix_count = (uint64_t)input.header.n * input.header.n;
    parameter = (int)kernel_input.header.n;
    if (input.header.count != matrix_count ||
        kernel_input.header.count != (uint64_t)parameter * parameter ||
        parameter <= 0 || !(parameter & 1) || parameter > (int)input.header.n) {
        fprintf(stderr, "Erro: matriz ou kernel incompativel.\n");
        goto cleanup;
    }
    const real_t *input_values = input.values;
    const real_t *kernel_values = kernel_input.values;
    real_t selected_sum = (real_t)0;
#if OPERATION == CONV_LINEAR
    real_t *matrix = copy_values(input_values, matrix_count);
    real_t *kernel = copy_values(kernel_values, kernel_input.header.count);
    real_t *output = calloc((size_t)matrix_count, sizeof(real_t));
    if (!matrix || !kernel || !output) { fprintf(stderr, "Erro: falha de alocacao.\n"); free(matrix); free(kernel); free(output); goto cleanup; }
#if TECHNIQUE == APPROXIMATE
    for (int ki = 0; ki < parameter; ki += 2)
        for (int kj = 0; kj < parameter; kj += 2) selected_sum += kernel[ki * parameter + kj];
    if (selected_sum == (real_t)0) { fprintf(stderr, "Erro: soma selecionada do kernel igual a zero.\n"); goto linear_free; }
    for (int ki = 0; ki < parameter; ki += 2)
        for (int kj = 0; kj < parameter; kj += 2) kernel[ki * parameter + kj] /= selected_sum;
#endif
    operation(output, matrix, (int)input.header.n, kernel, parameter, selected_sum);
    for (uint64_t i = 0; i < matrix_count; i++) checksum += output[i];
    if (!isfinite(checksum)) { fprintf(stderr, "Erro: resultado nao finito.\n"); goto linear_free; }
    if (dump_path && !dump_as_double(dump_path, output, matrix_count, IS_FLOAT)) { fprintf(stderr, "Erro: falha ao gravar saida.\n"); goto linear_free; }
    printf("Program=%s Operation=conv_linear Precision=%s Technique=%s N=%u Dist=%d Parameter=%d Seed=%u\n",
           PROGRAM_NAME, PRECISION_NAME, TECHNIQUE_NAME, input.header.n, input.header.dist, parameter, input.header.seed);
    printf("Checksum %.12f\n", (double)checksum); ok = 1;
linear_free:
    free(matrix); free(kernel); free(output);
#else
    real_t **matrix = allocate_rows((int)input.header.n, (int)input.header.n, 0);
    real_t **kernel = allocate_rows(parameter, parameter, 0);
    real_t **output = allocate_rows((int)input.header.n, (int)input.header.n, 1);
    if (!matrix || !kernel || !output) { fprintf(stderr, "Erro: falha de alocacao.\n"); goto malloc_free; }
    for (uint32_t i = 0; i < input.header.n; i++)
        for (uint32_t j = 0; j < input.header.n; j++) matrix[i][j] = input_values[(uint64_t)i * input.header.n + j];
    for (int i = 0; i < parameter; i++)
        for (int j = 0; j < parameter; j++) kernel[i][j] = kernel_values[i * parameter + j];
#if TECHNIQUE == APPROXIMATE
    for (int ki = 0; ki < parameter; ki += 2)
        for (int kj = 0; kj < parameter; kj += 2) selected_sum += kernel[ki][kj];
    if (selected_sum == (real_t)0) { fprintf(stderr, "Erro: soma selecionada do kernel igual a zero.\n"); goto malloc_free; }
    for (int ki = 0; ki < parameter; ki += 2)
        for (int kj = 0; kj < parameter; kj += 2) kernel[ki][kj] /= selected_sum;
#endif
    operation(output, matrix, (int)input.header.n, kernel, parameter, selected_sum);
    for (uint32_t i = 0; i < input.header.n; i++)
        for (uint32_t j = 0; j < input.header.n; j++) checksum += output[i][j];
    if (!isfinite(checksum)) { fprintf(stderr, "Erro: resultado nao finito.\n"); goto malloc_free; }
    if (dump_path) {
        real_t *flat = malloc((size_t)matrix_count * sizeof(real_t));
        if (!flat) { fprintf(stderr, "Erro: falha de alocacao.\n"); goto malloc_free; }
        for (uint32_t i = 0; i < input.header.n; i++)
            memcpy(flat + (uint64_t)i * input.header.n, output[i], (size_t)input.header.n * sizeof(real_t));
        if (!dump_as_double(dump_path, flat, matrix_count, IS_FLOAT)) {
            fprintf(stderr, "Erro: falha ao gravar saida.\n"); free(flat); goto malloc_free;
        }
        free(flat);
    }
    printf("Program=%s Operation=conv_malloc Precision=%s Technique=%s N=%u Dist=%d Parameter=%d Seed=%u\n",
           PROGRAM_NAME, PRECISION_NAME, TECHNIQUE_NAME, input.header.n, input.header.dist, parameter, input.header.seed);
    printf("Checksum %.12f\n", (double)checksum); ok = 1;
malloc_free:
    release_rows(matrix, (int)input.header.n); release_rows(kernel, parameter); release_rows(output, (int)input.header.n);
#endif
#endif
cleanup:
    input_release(&kernel_input);
    input_release(&input);
    return ok ? 0 : 1;
}
