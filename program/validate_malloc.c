#include <math.h>
#define main benchmark_main
#include "matrix_bench.c"
#undef main

int main(void) {
    real_t **matrix = allocate_rows(3, 3, 0);
    real_t **kernel = allocate_rows(3, 3, 0);
    real_t **output = allocate_rows(3, 3, 1);
    real_t selected = (real_t)0;
    double expected = TECHNIQUE == APPROXIMATE ? 7.0 : 285.0 / 45.0;
    double tolerance = IS_FLOAT ? 1e-5 : 1e-12;
    if (!matrix || !kernel || !output) return 1;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            matrix[i][j] = (real_t)(i * 3 + j + 1);
            kernel[i][j] = (real_t)(i * 3 + j + 1) / (real_t)45;
        }
    for (int i = 0; i < 3; i += 2)
        for (int j = 0; j < 3; j += 2) selected += kernel[i][j];
#if TECHNIQUE == APPROXIMATE
    for (int i = 0; i < 3; i += 2)
        for (int j = 0; j < 3; j += 2) kernel[i][j] /= selected;
#endif
    operation(output, matrix, 3, kernel, 3, selected);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double wanted = i == 1 && j == 1 ? expected : 0.0;
            if (fabs((double)output[i][j] - wanted) > tolerance) {
                fprintf(stderr, "%s: posicao (%d,%d) esperada %.12f, obtida %.12f\n",
                        PROGRAM_NAME, i, j, wanted, (double)output[i][j]);
                return 1;
            }
        }
    release_rows(matrix, 3); release_rows(kernel, 3); release_rows(output, 3);
    printf("OK: %s\n", PROGRAM_NAME);
    return 0;
}
