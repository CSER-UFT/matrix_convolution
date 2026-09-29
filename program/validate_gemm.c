#include <math.h>
#define main benchmark_main
#include "matrix_bench.c"
#undef main

int main(void) {
    real_t a[9] = {1,2,3,4,5,6,7,8,9};
    real_t b[9] = {9,8,7,6,5,4,3,2,1};
#if TECHNIQUE == APPROXIMATE
    double expected[9] = {27,21,15,81,66,51,135,111,87};
#else
    double expected[9] = {30,24,18,84,69,54,138,114,90};
#endif
    int blocks[] = {1,2,4};
    for (int choice = 0; choice < 3; choice++) {
        real_t output[9] = {0};
        operation(output, a, b, 3, blocks[choice], (real_t)1.5);
        for (int i = 0; i < 9; i++)
            if (fabs((double)output[i] - expected[i]) > 1e-6) {
                fprintf(stderr, "%s: bloco %d, posicao %d esperada %.12f, obtida %.12f\n",
                        PROGRAM_NAME, blocks[choice], i, expected[i], (double)output[i]);
                return 1;
            }
    }
    printf("OK: %s\n", PROGRAM_NAME);
    return 0;
}
