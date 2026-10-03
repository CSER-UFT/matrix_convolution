/*
 * validate.c: compara o kernel otimizado com uma implementação ingênua, em
 * double, para vários tamanhos (inclusive N não múltiplo do bloco), passos e
 * kernels (positivo e de soma zero). Compilado uma vez para cada binário.
 */
#define main benchmark_main
#include "matrix_bench.c"
#undef main

static uint64_t lcg_state = 12345u;
static double next_value(void) { /* uniforme em [-1, 1) */
    lcg_state = lcg_state * 6364136223846793005ull + 1442695040888963407ull;
    return (double)(lcg_state >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

static const double TOL = IS_FLOAT ? 2e-5 : 1e-12;
static int failures = 0;

static void check(double got, double want, double magnitude, const char *what,
                  int n, int param, int step, int i, int j) {
    if (fabs(got - want) > TOL * (magnitude + 1e-30)) {
        if (failures < 10)
            fprintf(stderr, "%s: %s N=%d param=%d step=%d (%d,%d) esperado %.15g obtido %.15g\n",
                    PROGRAM_NAME, what, n, param, step, i, j, want, got);
        failures++;
    }
}

#if OPERATION == GEMM_BLOCKED
static void test_gemm(int n, int bs, int step) {
    real_t *a = malloc((size_t)n * n * sizeof(real_t));
    real_t *b = malloc((size_t)n * n * sizeof(real_t));
    real_t *c = malloc((size_t)n * n * sizeof(real_t));
    for (int i = 0; i < n * n; i++) { a[i] = (real_t)next_value(); b[i] = (real_t)next_value(); }
    for (int i = 0; i < n * n; i++) c[i] = (real_t)99; /* o kernel deve zerar C */
    const double scale = (double)n / (double)kept_count(n, step);
    gemm_blocked(c, a, b, n, bs, step, (real_t)scale);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double want = 0.0, magnitude = 0.0;
            for (int k = 0; k < n; k += step) {
                double t = (double)((real_t)((double)a[i * n + k] * scale)) * (double)b[k * n + j];
                want += t; magnitude += fabs(t);
            }
            check((double)c[i * n + j], want, magnitude, "gemm", n, bs, step, i, j);
        }
    free(a); free(b); free(c);
}
#else
static void make_kernel(real_t *w, int k, int zero_sum) {
    if (!zero_sum) {
        double total = 0.0;
        for (int i = 0; i < k * k; i++) { w[i] = (real_t)(fabs(next_value()) + 0.01); total += (double)w[i]; }
        for (int i = 0; i < k * k; i++) w[i] = (real_t)((double)w[i] / total);
    } else { /* derivada horizontal: antissimétrica em j, soma zero */
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++) w[i * k + j] = (real_t)((j - k / 2) * (1 + (i < k - 1 - i ? i : k - 1 - i)));
    }
}

static void test_conv(int n, int k, int step, int zero_sum) {
    real_t *in = malloc((size_t)n * n * sizeof(real_t));
    real_t *w = malloc((size_t)k * k * sizeof(real_t));
    real_t *flat = calloc((size_t)n * n, sizeof(real_t));
    double l1_before = 0.0, l1_after = 0.0;
    for (int i = 0; i < n * n; i++) in[i] = (real_t)next_value();
    make_kernel(w, k, zero_sum);
    for (int i = 0; i < k * k; i++) l1_before += fabs((double)w[i]);
    if (!prepare_kernel(w, k, step)) { fprintf(stderr, "%s: prepare_kernel falhou k=%d step=%d\n", PROGRAM_NAME, k, step); failures++; goto done; }
    for (int i = 0; i < k * k; i++) l1_after += fabs((double)w[i]);
    check(l1_after, l1_before, l1_before, "norma L1 do kernel", n, k, step, 0, 0);
#if OPERATION == CONV_LINEAR
    conv_linear(flat, in, n, w, k, step);
#else
    real_t **m = allocate_rows(n, n, 0), **kw = allocate_rows(k, k, 0), **out = allocate_rows(n, n, 1);
    for (int i = 0; i < n; i++) memcpy(m[i], in + (size_t)i * n, (size_t)n * sizeof(real_t));
    for (int i = 0; i < k; i++) memcpy(kw[i], w + (size_t)i * k, (size_t)k * sizeof(real_t));
    conv_rows(out, m, n, kw, k, step);
    for (int i = 0; i < n; i++) memcpy(flat + (size_t)i * n, out[i], (size_t)n * sizeof(real_t));
    release_rows(m, n); release_rows(kw, k); release_rows(out, n);
#endif
    const int off = k / 2;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double want = 0.0, magnitude = 0.0;
            if (i >= off && i < n - off && j >= off && j < n - off)
                for (int ki = 0; ki < k; ki += step)
                    for (int kj = 0; kj < k; kj += step) {
                        double t = (double)in[(i - off + ki) * n + (j - off + kj)] * (double)w[ki * k + kj];
                        want += t; magnitude += fabs(t);
                    }
            check((double)flat[i * n + j], want, magnitude, "conv", n, k, step, i, j);
        }
done:
    free(in); free(w); free(flat);
}
#endif

int main(void) {
#if OPERATION == GEMM_BLOCKED
    const int sizes[] = {1, 7, 37, 64}, blocks[] = {1, 3, 8, 16, 64}, steps[] = {1, 2, 3, 4};
    for (int s = 0; s < 4; s++)
        for (int bl = 0; bl < 5; bl++)
            for (int st = 0; st < 4; st++)
                if (steps[st] <= sizes[s]) test_gemm(sizes[s], blocks[bl], steps[st]);
#else
    const int sizes[] = {7, 37, 40}, kernels[] = {1, 3, 5, 7};
    for (int s = 0; s < 3; s++)
        for (int kk = 0; kk < 4; kk++)
            for (int step = 1; step <= kernels[kk]; step++)
                for (int zero_sum = 0; zero_sum < 2; zero_sum++)
                    if (kernels[kk] <= sizes[s] && !(zero_sum && kernels[kk] == 1))
                        test_conv(sizes[s], kernels[kk], step, zero_sum);
#endif
    if (failures) { fprintf(stderr, "FALHOU: %s (%d divergências)\n", PROGRAM_NAME, failures); return 1; }
    printf("OK: %s\n", PROGRAM_NAME);
    return 0;
}
