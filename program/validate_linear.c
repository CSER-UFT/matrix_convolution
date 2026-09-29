#include <math.h>
#define main benchmark_main
#include "matrix_bench.c"
#undef main

int main(void) {
    real_t matrix[9] = {1,2,3,4,5,6,7,8,9};
    real_t kernel[9];
    real_t output[9] = {0};
    real_t selected = (real_t)0;
    double expected = TECHNIQUE == APPROXIMATE ? 7.0 : 285.0 / 45.0;
    double tolerance = IS_FLOAT ? 1e-5 : 1e-12;
    for (int i = 0; i < 9; i++) kernel[i] = (real_t)(i + 1) / (real_t)45;
    for (int i = 0; i < 3; i += 2)
        for (int j = 0; j < 3; j += 2) selected += kernel[i * 3 + j];
#if TECHNIQUE == APPROXIMATE
    for (int i = 0; i < 3; i += 2)
        for (int j = 0; j < 3; j += 2) kernel[i * 3 + j] /= selected;
#endif
    operation(output, matrix, 3, kernel, 3, selected);
    for (int i = 0; i < 9; i++) {
        double wanted = i == 4 ? expected : 0.0;
        if (fabs((double)output[i] - wanted) > tolerance) {
            fprintf(stderr, "%s: posicao %d esperada %.12f, obtida %.12f\n",
                    PROGRAM_NAME, i, wanted, (double)output[i]);
            return 1;
        }
    }
    printf("OK: %s\n", PROGRAM_NAME);
    return 0;
}
